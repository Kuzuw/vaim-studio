// VAIM Studio.exe - starts the Python app without a console window.
package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"unsafe"
)

func msg(text string) {
	user32 := syscall.NewLazyDLL("user32.dll")
	t, _ := syscall.UTF16PtrFromString(text)
	c, _ := syscall.UTF16PtrFromString("VAIM Studio")
	user32.NewProc("MessageBoxW").Call(0, uintptr(unsafe.Pointer(t)), uintptr(unsafe.Pointer(c)), 0x10)
}

func main() {
	exe, err := os.Executable()
	if err != nil {
		msg("Could not locate VAIM Studio: " + err.Error())
		return
	}
	dir := filepath.Dir(exe)
	pyw := filepath.Join(dir, ".venv", "Scripts", "pythonw.exe")
	script := filepath.Join(dir, "launch.pyw")
	if _, err := os.Stat(pyw); err != nil {
		msg("VAIM Studio isn't fully installed (Python environment missing).\n\nPlease run VAIM-Studio-Setup.exe again.")
		return
	}
	cmd := exec.Command(pyw, script)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "PYTHONNOUSERSITE=1")
	if err := cmd.Start(); err != nil {
		msg("Could not start VAIM Studio:\n" + err.Error())
	}
}
