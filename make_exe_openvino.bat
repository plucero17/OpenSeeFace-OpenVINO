@echo off
setlocal

echo Creating OpenVINO virtual environment
python -m venv venv-openvino
if errorlevel 1 goto :error

call venv-openvino\Scripts\activate.bat
if errorlevel 1 goto :error

python -m pip install --upgrade pip wheel
python -m pip install numpy==1.26.4 opencv-python==4.10.0.84 pillow onnxruntime openvino pyinstaller
if errorlevel 1 goto :error

for /f "usebackq tokens=*" %%i in (`python -c "import sysconfig; print(sysconfig.get_path('purelib'))"`) do set SITEPACKAGES=%%i
set OV_LIB_DIR=%SITEPACKAGES%\openvino\libs
set VC_RUNTIME_DIR=%SystemRoot%\System32
if not exist "%OV_LIB_DIR%" (
    echo Could not locate OpenVINO libraries under %OV_LIB_DIR%
    goto :error
)
if not exist "%VC_RUNTIME_DIR%\msvcp140.dll" (
    echo Could not locate Visual C++ runtime libraries under %VC_RUNTIME_DIR%
    goto :error
)

echo Building facetracker_openvino bundle
pyinstaller facetracker.py --clean --onedir ^
    --name facetracker_openvino ^
    --add-data ov-models;ov-models ^
    --add-binary "%OV_LIB_DIR%\*.dll";openvino\libs ^
    --add-binary dshowcapture\*.dll;. ^
    --add-binary escapi\*.dll;. ^
    --add-binary "%VC_RUNTIME_DIR%\msvcp140.dll";. ^
    --add-binary "%VC_RUNTIME_DIR%\vcomp140.dll";. ^
    --add-binary "%VC_RUNTIME_DIR%\concrt140.dll";. ^
    --add-binary "%VC_RUNTIME_DIR%\vccorlib140.dll";. ^
    --add-binary run.bat;.
if errorlevel 1 goto :error

echo Done. Check dist\facetracker_openvino for the executable.
exit /b 0

:error
echo Build failed. See log above.
exit /b 1
