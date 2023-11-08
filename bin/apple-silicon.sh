# check for apple silicon homebrew
if [ -d "/opt/homebrew/lib" ]; then
  export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_FALLBACK_LIBRARY_PATH"
fi
