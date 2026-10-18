// VAIM-Studio-Setup.exe - installs VAIM Studio for the current user (no admin needed).
//
// What it does:
//   1. extracts the app (embedded payload.zip) to %LOCALAPPDATA%\Programs\VAIM Studio
//   2. detects the NVIDIA GPU and picks the matching CUDA build of PyTorch
//   3. fetches the uv package manager, which installs a private Python 3.12 + PyTorch
//   4. runs the app's self-test
//   5. adds Desktop / Start menu shortcuts and an "Apps & features" uninstall entry
// Run with --uninstall (or as uninstall.exe) to remove it.
package main

import (
	"archive/zip"
	"bufio"
	"bytes"
	_ "embed"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
	"unsafe"
)

//go:embed payload.zip
var payload []byte

// appVersion is overridden at build time: -ldflags "-X main.appVersion=1.2.3"
var appVersion = "dev"

const (
	appName    = "VAIM Studio"

	regKey     = `HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\VAIMStudio`
	torchVer   = "2.7.1"
	needGB     = 7
	uvURL      = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
	marker     = "vaim_studio" // folder that proves a directory is a VAIM Studio install
)

var (
	kernel32 = syscall.NewLazyDLL("kernel32.dll")
	in       = bufio.NewReader(os.Stdin)
)

// ---------------------------------------------------------------- console helpers

func enableColors() {
	h, _ := syscall.GetStdHandle(syscall.STD_OUTPUT_HANDLE)
	var mode uint32
	kernel32.NewProc("GetConsoleMode").Call(uintptr(h), uintptr(unsafe.Pointer(&mode)))
	kernel32.NewProc("SetConsoleMode").Call(uintptr(h), uintptr(mode|0x0004)) // VT processing
	t, _ := syscall.UTF16PtrFromString(appName + " Setup")
	kernel32.NewProc("SetConsoleTitleW").Call(uintptr(unsafe.Pointer(t)))
}

const (
	cReset  = "\x1b[0m"
	cDim    = "\x1b[90m"
	cViolet = "\x1b[95m"
	cGreen  = "\x1b[92m"
	cYellow = "\x1b[93m"
	cRed    = "\x1b[91m"
	cCyan   = "\x1b[96m"
	cBold   = "\x1b[1m"
)

func say(color, format string, a ...any) { fmt.Printf(color+format+cReset+"\n", a...) }

func ask(prompt string) string {
	fmt.Print(prompt)
	line, _ := in.ReadString('\n')
	return strings.TrimSpace(line)
}

func yes(prompt string, def bool) bool {
	a := strings.ToLower(ask(prompt))
	if a == "" {
		return def
	}
	return strings.HasPrefix(a, "y") || strings.HasPrefix(a, "e") // also Turkish "evet"
}

func fail(format string, a ...any) {
	fmt.Println()
	say(cRed+cBold, "  Setup failed: "+format, a...)
	fmt.Println()
	ask("  Press Enter to close ")
	os.Exit(1)
}

func step(n, total int, text string) {
	fmt.Println()
	say(cCyan+cBold, "  [%d/%d] %s", n, total, text)
}

// ---------------------------------------------------------------- system helpers

func freeGB(dir string) float64 {
	for d := dir; ; d = filepath.Dir(d) {
		if _, err := os.Stat(d); err == nil {
			p, _ := syscall.UTF16PtrFromString(d)
			var free, total, totalFree uint64
			r, _, _ := kernel32.NewProc("GetDiskFreeSpaceExW").Call(uintptr(unsafe.Pointer(p)),
				uintptr(unsafe.Pointer(&free)), uintptr(unsafe.Pointer(&total)), uintptr(unsafe.Pointer(&totalFree)))
			if r == 0 {
				return -1
			}
			return float64(free) / (1 << 30)
		}
		if filepath.Dir(d) == d {
			return -1
		}
	}
}

func run(dir string, env []string, name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	cmd.Stdout, cmd.Stderr, cmd.Stdin = os.Stdout, os.Stderr, nil
	cmd.Env = append(os.Environ(), env...)
	return cmd.Run()
}

func output(name string, args ...string) (string, error) {
	cmd := exec.Command(name, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}
	b, err := cmd.Output()
	return string(b), err
}

func powershell(script string) error {
	cmd := exec.Command("powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script)
	cmd.Stdout, cmd.Stderr = io.Discard, os.Stderr
	return cmd.Run()
}

func psQuote(s string) string { return "'" + strings.ReplaceAll(s, "'", "''") + "'" }

type progressWriter struct {
	total, done int64
	last        time.Time
	label       string
}

func (p *progressWriter) Write(b []byte) (int, error) {
	p.done += int64(len(b))
	if time.Since(p.last) > 150*time.Millisecond || p.done == p.total {
		p.last = time.Now()
		if p.total > 0 {
			pct := float64(p.done) / float64(p.total)
			bar := strings.Repeat("#", int(pct*30)) + strings.Repeat(".", 30-int(pct*30))
			fmt.Printf("\r    %s [%s] %3.0f%%  %.1f / %.1f MB ", p.label, bar, pct*100, float64(p.done)/1e6, float64(p.total)/1e6)
		} else {
			fmt.Printf("\r    %s %.1f MB ", p.label, float64(p.done)/1e6)
		}
	}
	return len(b), nil
}

func download(url, dest, label string) error {
	var lastErr error
	for attempt := 1; attempt <= 3; attempt++ {
		resp, err := http.Get(url)
		if err == nil && resp.StatusCode != 200 {
			err = fmt.Errorf("HTTP %d", resp.StatusCode)
			resp.Body.Close()
		}
		if err != nil {
			lastErr = err
			time.Sleep(time.Duration(attempt) * 2 * time.Second)
			continue
		}
		f, err := os.Create(dest)
		if err != nil {
			resp.Body.Close()
			return err
		}
		_, err = io.Copy(f, io.TeeReader(resp.Body, &progressWriter{total: resp.ContentLength, label: label}))
		resp.Body.Close()
		f.Close()
		fmt.Println()
		if err == nil {
			return nil
		}
		lastErr = err
	}
	return lastErr
}

func unzip(data []byte, dest string, only func(string) bool) error {
	r, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return err
	}
	for _, f := range r.File {
		if only != nil && !only(f.Name) {
			continue
		}
		p := filepath.Join(dest, filepath.FromSlash(f.Name))
		if !strings.HasPrefix(p, filepath.Clean(dest)+string(os.PathSeparator)) {
			return fmt.Errorf("bad path in archive: %s", f.Name)
		}
		if f.FileInfo().IsDir() {
			os.MkdirAll(p, 0o755)
			continue
		}
		os.MkdirAll(filepath.Dir(p), 0o755)
		rc, err := f.Open()
		if err != nil {
			return err
		}
		out, err := os.Create(p)
		if err != nil {
			rc.Close()
			return fmt.Errorf("%v (is VAIM Studio still running? close it and retry)", err)
		}
		_, err = io.Copy(out, rc)
		rc.Close()
		out.Close()
		if err != nil {
			return err
		}
	}
	return nil
}

func isInstall(dir string) bool {
	st, err := os.Stat(filepath.Join(dir, marker))
	return err == nil && st.IsDir()
}

func isEmptyDir(dir string) bool {
	entries, err := os.ReadDir(dir)
	return err == nil && len(entries) == 0
}

// ---------------------------------------------------------------- GPU detection

type gpuInfo struct {
	name  string
	memGB float64
	cc    float64 // compute capability, 0 if unknown
}

func detectGPU() *gpuInfo {
	smi := "nvidia-smi"
	if _, err := exec.LookPath(smi); err != nil {
		alt := `C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe`
		if _, err := os.Stat(alt); err != nil {
			return nil
		}
		smi = alt
	}
	out, err := output(smi, "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits")
	if err != nil { // older drivers don't know compute_cap
		out, err = output(smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits")
		if err != nil {
			return nil
		}
	}
	line := strings.TrimSpace(strings.SplitN(strings.TrimSpace(out), "\n", 2)[0])
	parts := strings.Split(line, ",")
	g := &gpuInfo{name: strings.TrimSpace(parts[0])}
	if len(parts) > 1 {
		mb, _ := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
		g.memGB = mb / 1024
	}
	if len(parts) > 2 {
		g.cc, _ = strconv.ParseFloat(strings.TrimSpace(parts[2]), 64)
	}
	return g
}

// ---------------------------------------------------------------- MSVC runtime

// PyTorch 2.7 is built with Visual Studio 2022 17.10+, whose C++ runtime changed std::mutex.
// An older msvcp140.dll in System32 can crash PyTorch with heap corruption (0xc0000374).
const vcMin = 14.40

func vcRuntimeVersion() float64 {
	out, err := output("powershell", "-NoProfile", "-Command",
		`$v=(Get-Item "$env:WINDIR\System32\msvcp140.dll" -ErrorAction SilentlyContinue).VersionInfo; if ($v) { "{0}.{1:D2}" -f $v.FileMajorPart, $v.FileMinorPart }`)
	if err != nil {
		return 0
	}
	v, _ := strconv.ParseFloat(strings.TrimSpace(out), 64)
	return v
}

func ensureVCRuntime(tools string) {
	v := vcRuntimeVersion()
	if v >= vcMin {
		say(cDim, "    Microsoft C++ runtime %.2f - OK", v)
		return
	}
	if v == 0 {
		say(cYellow, "    Microsoft C++ runtime not found - installing it (Windows will ask for permission).")
	} else {
		say(cYellow, "    Microsoft C++ runtime %.2f is too old for PyTorch - updating it (Windows will ask for permission).", v)
	}
	exe := filepath.Join(tools, "vc_redist.x64.exe")
	if err := download("https://aka.ms/vs/17/release/vc_redist.x64.exe", exe, "vc_redist"); err != nil {
		say(cYellow, "    Could not download it (%v). Get it from https://aka.ms/vs/17/release/vc_redist.x64.exe", err)
		return
	}
	err := powershell(fmt.Sprintf(`$p = Start-Process -FilePath %s -ArgumentList '/install','/quiet','/norestart' -Verb RunAs -Wait -PassThru; exit $p.ExitCode`, psQuote(exe)))
	os.Remove(exe)
	if nv := vcRuntimeVersion(); nv >= vcMin {
		say(cGreen, "    Microsoft C++ runtime updated to %.2f", nv)
	} else {
		say(cYellow, "    The C++ runtime update did not complete (%v). If VAIM Studio crashes, install", err)
		say(cYellow, "    https://aka.ms/vs/17/release/vc_redist.x64.exe manually.")
	}
}

func cudaIndex(g *gpuInfo) string {
	if g == nil {
		return "cpu"
	}
	if g.cc >= 12.0 || strings.Contains(strings.ToUpper(g.name), "RTX 50") {
		return "cu128" // Blackwell (RTX 50xx) needs CUDA 12.8 builds
	}
	if g.cc > 0 && g.cc < 5.0 {
		return "cpu"
	}
	return "cu126" // GTX 9xx / 10xx / 16xx, RTX 20/30/40
}

// ---------------------------------------------------------------- install

func install() {
	say(cViolet+cBold, "\n  VAIM Studio  %s", appVersion)
	say(cDim, "  vintage ai model studio  -  BigGAN + CLIP image dreaming")
	say(cDim, "  ------------------------------------------------------")
	fmt.Println()

	def := filepath.Join(os.Getenv("LOCALAPPDATA"), "Programs", appName)
	if out, err := output("reg", "query", regKey, "/v", "InstallLocation"); err == nil { // update in place
		for _, line := range strings.Split(out, "\n") {
			if i := strings.Index(line, "REG_SZ"); i >= 0 {
				if p := strings.TrimSpace(line[i+6:]); isInstall(p) {
					def = p
				}
			}
		}
	}
	dir := def
	fmt.Printf("  Install folder:\n    %s\n", def)
	if a := ask("  Press Enter to accept, or type another folder: "); a != "" {
		dir = filepath.Clean(strings.Trim(a, `"`))
	}
	if !filepath.IsAbs(dir) {
		fail("please enter a full path like D:\\Apps\\VAIM Studio")
	}
	// never spill into a non-empty folder that isn't ours (e.g. "D:\")
	if st, err := os.Stat(dir); err == nil && st.IsDir() && !isInstall(dir) && !isEmptyDir(dir) {
		dir = filepath.Join(dir, appName)
	}
	updating := isInstall(dir)

	g := detectGPU()
	idx := cudaIndex(g)
	fmt.Println()
	if g != nil {
		cc := "?"
		if g.cc > 0 {
			cc = fmt.Sprintf("%.1f", g.cc)
		}
		say(cGreen, "  GPU: %s  (%.1f GB VRAM, compute %s)", g.name, g.memGB, cc)
		if g.memGB > 0 && g.memGB < 6 {
			say(cDim, "  Small-VRAM card: VAIM Studio will tune memory savers for it automatically.")
		}
	} else {
		say(cYellow, "  No NVIDIA GPU found (nvidia-smi missing). If you have one, install the latest")
		say(cYellow, "  driver from nvidia.com first. Continuing with a CPU install (very slow).")
	}
	if idx == "cpu" && g != nil {
		say(cYellow, "  This GPU is too old for current PyTorch - installing the CPU build.")
	}
	free := freeGB(dir)
	if !updating && free >= 0 && free < needGB {
		fail("only %.1f GB free on that drive; about %d GB is needed.", free, needGB)
	}
	fmt.Println()
	if updating {
		say(cBold, "  Updating the existing install in %s", dir)
	} else {
		say(cBold, "  Installing to %s", dir)
	}
	say(cDim, "  Downloads: Python + PyTorch (~2.6 GB) now, models (~700 MB) on first use.")
	if !yes("  Continue? [Y/n] ", true) {
		os.Exit(0)
	}

	const N = 7
	tools := filepath.Join(dir, ".tools")
	env := []string{
		"UV_PYTHON_INSTALL_DIR=" + filepath.Join(tools, "python"),
		"UV_CACHE_DIR=" + filepath.Join(tools, "cache"),
		"UV_LINK_MODE=copy",
		"UV_HTTP_TIMEOUT=300",
		"PYTHONNOUSERSITE=1",
	}

	// 1. files
	step(1, N, "Copying app files")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		fail("cannot create %s: %v", dir, err)
	}
	for _, old := range []string{marker, "tools"} { // clear stale code from older versions
		os.RemoveAll(filepath.Join(dir, old))
	}
	if err := unzip(payload, dir, nil); err != nil {
		fail("%v", err)
	}

	// 2. C++ runtime
	step(2, N, "Checking the Microsoft C++ runtime")
	os.MkdirAll(tools, 0o755)
	ensureVCRuntime(tools)

	// 3. uv
	step(3, N, "Getting the uv package manager")
	uv := filepath.Join(tools, "uv.exe")
	if _, err := os.Stat(uv); err != nil {
		os.MkdirAll(tools, 0o755)
		zp := filepath.Join(tools, "uv.zip")
		if err := download(uvURL, zp, "uv"); err != nil {
			fail("could not download uv: %v\n  Check your internet connection and try again.", err)
		}
		data, _ := os.ReadFile(zp)
		if err := unzip(data, tools, func(n string) bool { return strings.HasSuffix(n, ".exe") }); err != nil {
			fail("could not unpack uv: %v", err)
		}
		os.Remove(zp)
		// some uv zips nest the exe in a folder
		if _, err := os.Stat(uv); err != nil {
			filepath.Walk(tools, func(p string, info os.FileInfo, _ error) error {
				if info != nil && strings.EqualFold(info.Name(), "uv.exe") && p != uv {
					os.Rename(p, uv)
				}
				return nil
			})
		}
	} else {
		say(cDim, "    already present")
	}

	// 4. python
	step(4, N, "Creating a private Python 3.12")
	if err := run(dir, env, uv, "venv", ".venv", "--python", "3.12", "--allow-existing"); err != nil {
		fail("could not create the Python environment (%v)", err)
	}
	py := filepath.Join(dir, ".venv", "Scripts", "python.exe")

	// 5. torch
	step(5, N, fmt.Sprintf("Installing PyTorch %s (%s) - the big download, ~2.5 GB", torchVer, idx))
	if err := run(dir, env, uv, "pip", "install", "--python", py, "torch=="+torchVer,
		"--index-url", "https://download.pytorch.org/whl/"+idx); err != nil {
		fail("PyTorch install failed (%v). Re-run setup to resume.", err)
	}

	// 6. the rest
	step(6, N, "Installing the app's other packages")
	if err := run(dir, env, uv, "pip", "install", "--python", py, "-r", "requirements.txt"); err != nil {
		fail("package install failed (%v). Re-run setup to resume.", err)
	}

	// 7. check + integrate
	step(7, N, "Checking the install")
	check := "import torch; ok = torch.cuda.is_available(); " +
		"print('    PyTorch', torch.__version__, ('| CUDA OK on ' + torch.cuda.get_device_name(0)) if ok else '| CPU only')"
	run(dir, env, py, "-c", check)
	if err := run(dir, env, py, filepath.Join("tools", "selftest.py")); err != nil {
		say(cYellow, "    The self-test reported a problem (see above). The app may still work.")
	}

	launcher := filepath.Join(dir, appName+".exe")
	icon := filepath.Join(dir, marker, "assets", "icon.ico")
	ps := fmt.Sprintf(`$s = New-Object -ComObject WScript.Shell
foreach ($d in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {
  $l = $s.CreateShortcut((Join-Path $d '%s.lnk'))
  $l.TargetPath = %s; $l.WorkingDirectory = %s; $l.IconLocation = %s
  $l.Description = 'Vintage AI Model Studio'; $l.Save() }`, appName, psQuote(launcher), psQuote(dir), psQuote(icon))
	if err := powershell(ps); err != nil {
		say(cYellow, "    (could not create shortcuts: %v)", err)
	}

	self, _ := os.Executable()
	unin := filepath.Join(dir, "uninstall.exe")
	if !strings.EqualFold(filepath.Clean(self), unin) {
		if data, err := os.ReadFile(self); err == nil {
			os.WriteFile(unin, data, 0o755)
		}
	}
	for _, kv := range [][2]string{
		{"DisplayName", appName}, {"DisplayVersion", appVersion}, {"Publisher", "VAIM Studio"},
		{"DisplayIcon", icon}, {"InstallLocation", dir}, {"UninstallString", `"` + unin + `" --uninstall`},
	} {
		output("reg", "add", regKey, "/v", kv[0], "/t", "REG_SZ", "/d", kv[1], "/f")
	}
	output("reg", "add", regKey, "/v", "NoModify", "/t", "REG_DWORD", "/d", "1", "/f")
	output("reg", "add", regKey, "/v", "NoRepair", "/t", "REG_DWORD", "/d", "1", "/f")
	output("reg", "add", regKey, "/v", "EstimatedSize", "/t", "REG_DWORD", "/d", "5200000", "/f")

	os.RemoveAll(filepath.Join(tools, "cache")) // the download cache would double the disk use

	fmt.Println()
	say(cGreen+cBold, "  Done! VAIM Studio is on your Desktop and in the Start menu.")
	say(cDim, "  The first Dream downloads BigGAN, CLIP and Real-ESRGAN (~700 MB, once).")
	say(cDim, "  Your images go to Pictures\\VAIM Studio.")
	fmt.Println()
	if yes("  Launch VAIM Studio now? [Y/n] ", true) {
		cmd := exec.Command(launcher)
		cmd.Dir = dir
		cmd.Start()
	}
}

// ---------------------------------------------------------------- uninstall

func uninstall(dir string) {
	say(cViolet+cBold, "\n  VAIM Studio - uninstall\n")
	if !isInstall(dir) {
		fail("%s does not look like a VAIM Studio install; not touching it.", dir)
	}
	say(cBold, "  This removes VAIM Studio from:\n    %s", dir)
	say(cDim, "  Your images in Pictures\\VAIM Studio are kept.")
	if !yes("  Uninstall? [y/N] ", false) {
		os.Exit(0)
	}
	time.Sleep(800 * time.Millisecond) // let the copy we were started from exit
	powershell(fmt.Sprintf(`foreach ($d in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {
  Remove-Item -LiteralPath (Join-Path $d '%s.lnk') -ErrorAction SilentlyContinue }`, appName))
	output("reg", "delete", regKey, "/f")
	var err error
	for i := 0; i < 10; i++ {
		if err = os.RemoveAll(dir); err == nil {
			break
		}
		time.Sleep(time.Second)
	}
	if err != nil {
		say(cYellow, "  Some files could not be removed (is VAIM Studio still open?): %v", err)
	}
	data := filepath.Join(os.Getenv("LOCALAPPDATA"), appName)
	if _, e := os.Stat(data); e == nil {
		if yes("  Also delete the downloaded models and settings (~1 GB)? [y/N] ", false) {
			os.RemoveAll(data)
		}
	}
	say(cGreen, "\n  VAIM Studio has been removed.")
	ask("  Press Enter to close ")
	// delete this temporary copy of the uninstaller once we've exited
	self, _ := os.Executable()
	cmd := exec.Command(os.Getenv("ComSpec"))
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true,
		CmdLine: `cmd /c ping -n 3 127.0.0.1 >nul & del /q "` + self + `"`}
	cmd.Start()
}

func main() {
	enableColors()
	args := os.Args[1:]
	self, _ := os.Executable()
	selfDir := filepath.Dir(self)
	if len(args) >= 2 && args[0] == "--uninstall-from" {
		uninstall(args[1])
		return
	}
	if (len(args) > 0 && args[0] == "--uninstall") || strings.EqualFold(filepath.Base(self), "uninstall.exe") {
		// can't delete our own folder while running from it: continue from a temp copy
		tmp := filepath.Join(os.TempDir(), fmt.Sprintf("vaim-uninstall-%d.exe", time.Now().UnixNano()))
		data, err := os.ReadFile(self)
		if err == nil {
			err = os.WriteFile(tmp, data, 0o755)
		}
		if err != nil {
			fail("could not start the uninstaller: %v", err)
		}
		cmd := exec.Command(tmp, "--uninstall-from", selfDir)
		cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: 0x00000010} // CREATE_NEW_CONSOLE
		if err := cmd.Start(); err != nil {
			fail("could not start the uninstaller: %v", err)
		}
		return
	}
	install()
}
