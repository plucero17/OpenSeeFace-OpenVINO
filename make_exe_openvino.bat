@echo off
setlocal

echo Creating OpenVINO virtual environment
python -m venv venv-openvino
if errorlevel 1 goto :error

call venv-openvino\Scripts\activate.bat
if errorlevel 1 goto :error

pip install --upgrade pip wheel
pip install numpy==1.23.0 opencv-python==4.5.4.60 pillow onnxruntime openvino pyinstaller
if errorlevel 1 goto :error

for /f "usebackq tokens=*" %%i in (`python -c "import sysconfig; print(sysconfig.get_path('purelib'))"`) do set SITEPACKAGES=%%i
set OV_LIB_DIR=%SITEPACKAGES%\openvino\libs
if not exist "%OV_LIB_DIR%" (
    echo Could not locate OpenVINO libraries under %OV_LIB_DIR%
    goto :error
)

echo Building facetracker_openvino bundle
pyinstaller facetracker.py --clean --onedir ^
    --name facetracker_openvino ^
    --add-data ov-models;ov-models ^
    --add-data models;models ^
    --add-binary "%OV_LIB_DIR%\*.dll";openvino\libs ^
    --add-binary dshowcapture\*.dll;. ^
    --add-binary escapi\*.dll;. ^
    --add-binary msvcp140.dll;. ^
    --add-binary vcomp140.dll;. ^
    --add-binary concrt140.dll;. ^
    --add-binary vccorlib140.dll;. ^
    --add-binary run.bat;.
if errorlevel 1 goto :error

echo Done. Check dist\facetracker_openvino for the executable.
exit /b 0

:error
echo Build failed. See log above.
exit /b 1
