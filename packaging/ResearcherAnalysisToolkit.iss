#define MyAppName "Researcher Analysis Toolkit"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "TEAM Clean Energy Research Center"
#define MyAppURL "https://github.com/Milad-Behroozi"
#define MyAppExeName "ResearcherAnalysisToolkit.exe"

[Setup]
AppId={{7E78594A-492A-49A8-9A5C-89E5952821B4}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=output
OutputBaseFilename=Researcher_Analysis_Toolkit_v1.0.0_Setup
SetupIconFile=..\Researcherlogo.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
RestartIfNeededByRun=no
UsePreviousAppDir=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "..\dist\ResearcherAnalysisToolkit\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\python-3.11.9-amd64.exe"; Parameters: "/quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_launcher=1 Include_test=0 Include_doc=0 Include_debug=0 Shortcuts=0"; StatusMsg: "Installing the supported Python 3.11 runtime..."; Flags: waituntilterminated; Check: not Python311Present
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent

[Code]
function Python311Present: Boolean;
var
  InstallPath: String;
begin
  Result := FileExists(ExpandConstant('{localappdata}\Programs\Python\Python311\python.exe'));

  if not Result then
    Result := RegQueryStringValue(
      HKCU,
      'Software\Python\PythonCore\3.11\InstallPath',
      '',
      InstallPath
    );

  if not Result then
    Result := RegQueryStringValue(
      HKLM64,
      'Software\Python\PythonCore\3.11\InstallPath',
      '',
      InstallPath
    );

  if not Result then
    Result := RegQueryStringValue(
      HKLM32,
      'Software\Python\PythonCore\3.11\InstallPath',
      '',
      InstallPath
    );
end;

