'Option Explicit

Dim shell, fso, projectDir, pythonwPath, scriptPath, command
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
projectDir = fso.GetParentFolderName(WScript.ScriptFullName)
scriptPath = fso.BuildPath(projectDir, "run_app.py")
shell.CurrentDirectory = projectDir

' Prefer the project's virtual environment when it exists.
pythonwPath = fso.BuildPath(projectDir, ".venv\Scripts\pythonw.exe")
If Not fso.FileExists(pythonwPath) Then
    pythonwPath = "pythonw.exe" ' Uses the Python installation available on PATH.
End If

command = Chr(34) & pythonwPath & Chr(34) & " " & Chr(34) & scriptPath & Chr(34)
On Error Resume Next
shell.Run command, 0, False
If Err.Number <> 0 Then
    MsgBox "Could not start Python without a console window." & vbCrLf & _
           "Install Python and make sure pythonw.exe is available on PATH." & vbCrLf & vbCrLf & _
           "You can still use run_app_debug.bat to see startup errors.", _
           vbCritical, "FraudShield launcher"
End If
On Error GoTo 0
