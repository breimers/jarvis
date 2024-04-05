#!/bin/bash
BIN_NAME=$(echo $1 | sed 's/\.[^.]*$//')
echo "***Compiling $BIN_NAME***"
echo "*** Clearing dist ***"
rm -rf dist
echo "*** Recreating dist env ***."
mkdir -p dist
echo "*** Building binary ***"
pyinstaller --onefile $1
echo "*** Signing code ***"
codesign --deep --force --options=runtime --entitlements '/Users/breimers/Workshop/breimers/jarvis/ui/macos/Jarvis macOS/Jarvis macOS/Jarvis_macOS.entitlements' --sign "937B257DF19133F12FC028BACC47B2BD7A2B964E" --timestamp dist/$BIN_NAME
echo "*** Done! ***"