#!/bin/bash
# Minimal debug script to diagnose missing commands/files in sandbox container
# Run this INSTEAD of start-with-nginx.sh to isolate environment issues
#
# Usage: Replace entrypoint with this script, or run manually:
#   docker run --rm -it <image> /debug-env.sh
#   enroot start <container> /debug-env.sh

echo "=== SANDBOX ENVIRONMENT DEBUG ==="
echo "Timestamp: $(date 2>&1 || echo 'date FAILED')"
echo "Hostname: $(hostname 2>&1 || echo 'hostname FAILED')"
echo ""

# 1. Check PATH
echo "=== PATH ==="
echo "PATH=$PATH"
echo ""

# 2. Check critical directories exist
echo "=== Critical Directories ==="
for dir in /bin /usr/bin /sbin /usr/sbin /lib /lib64 /usr/lib; do
    if [ -d "$dir" ]; then
        count=$(ls -1 "$dir" 2>/dev/null | wc -l)
        echo "$dir: EXISTS ($count files)"
    else
        echo "$dir: MISSING!"
    fi
done
echo ""

# 3. Check /dev/null and other special files
echo "=== Special Files ==="
for f in /dev/null /dev/zero /dev/urandom /dev/stdout /dev/stderr; do
    if [ -e "$f" ]; then
        echo "$f: EXISTS ($(ls -la "$f" 2>&1))"
    else
        echo "$f: MISSING!"
    fi
done
echo ""

# 4. Check critical commands (both via PATH and absolute)
echo "=== Critical Commands ==="
commands="date rm mkdir cat ls echo grep awk sed curl python3 nginx uwsgi ss"
for cmd in $commands; do
    # Try via PATH
    if command -v "$cmd" >/dev/null 2>&1; then
        path=$(command -v "$cmd")
        echo "$cmd: FOUND at $path"
    else
        echo "$cmd: NOT IN PATH"
        # Try common absolute paths
        for prefix in /bin /usr/bin /sbin /usr/sbin; do
            if [ -x "$prefix/$cmd" ]; then
                echo "  -> But exists at $prefix/$cmd"
                break
            fi
        done
    fi
done
echo ""

# 5. Check ld.so.preload (this can break everything)
echo "=== ld.so.preload Check ==="
if [ -f /etc/ld.so.preload ]; then
    echo "/etc/ld.so.preload EXISTS:"
    cat /etc/ld.so.preload
    echo ""
    echo "WARNING: ld.so.preload can break commands if the library has issues!"
else
    echo "/etc/ld.so.preload: not present (good)"
fi
echo ""

# 6. Check if libblock_network.so exists and is valid
echo "=== Network Blocking Library ==="
BLOCK_LIB="/usr/lib/libblock_network.so"
if [ -f "$BLOCK_LIB" ]; then
    echo "$BLOCK_LIB: EXISTS"
    file "$BLOCK_LIB" 2>&1 || echo "  file command failed"
    ldd "$BLOCK_LIB" 2>&1 || echo "  ldd command failed"
else
    echo "$BLOCK_LIB: not present"
fi
echo ""

# 7. Check environment variables that might affect behavior
echo "=== Environment Variables ==="
echo "LD_PRELOAD=${LD_PRELOAD:-<not set>}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<not set>}"
echo "NEMO_SKILLS_SANDBOX_BLOCK_NETWORK=${NEMO_SKILLS_SANDBOX_BLOCK_NETWORK:-<not set>}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<not set>}"
echo "SLURM_NODEID=${SLURM_NODEID:-<not set>}"
echo ""

# 8. Check mounts
echo "=== Mount Points ==="
mount 2>&1 | head -20 || echo "mount command failed"
echo "..."
echo ""

# 9. Check /proc filesystem
echo "=== /proc Filesystem ==="
if [ -d /proc ]; then
    echo "/proc: EXISTS"
    if [ -f /proc/self/status ]; then
        echo "  PID: $$"
        grep -E "^(Name|Uid|Gid):" /proc/self/status 2>&1 || echo "  Cannot read /proc/self/status"
    fi
else
    echo "/proc: MISSING (container may be misconfigured)"
fi
echo ""

# 10. Try running commands that failed in the original script
echo "=== Reproducing Original Failures ==="
echo "Testing 'date +%s':"
result=$(date +%s 2>&1)
echo "  Result: $result"
echo "  Exit code: $?"

echo "Testing '/usr/bin/rm --version':"
result=$(/usr/bin/rm --version 2>&1)
echo "  Result: $result"
echo "  Exit code: $?"

echo "Testing 'echo test > /dev/null':"
echo test > /dev/null 2>&1
echo "  Exit code: $?"

echo "Testing 'mktemp -d':"
result=$(mktemp -d 2>&1)
echo "  Result: $result"
echo "  Exit code: $?"
[ -d "$result" ] && rmdir "$result" 2>/dev/null

echo ""

# 11. Check if this is a timing/race issue
echo "=== Timing Test ==="
echo "Running date 10 times with 0.1s delay:"
for i in 1 2 3 4 5 6 7 8 9 10; do
    result=$(date +%s 2>&1)
    if [ $? -eq 0 ]; then
        echo -n "."
    else
        echo ""
        echo "FAILED on iteration $i: $result"
        break
    fi
    sleep 0.1 2>/dev/null || true
done
echo " done"
echo ""

# 12. Summary
echo "=== SUMMARY ==="
errors=0

# Check critical items
command -v date >/dev/null 2>&1 || { echo "CRITICAL: 'date' not found"; errors=$((errors+1)); }
command -v rm >/dev/null 2>&1 || { echo "CRITICAL: 'rm' not found"; errors=$((errors+1)); }
[ -e /dev/null ] || { echo "CRITICAL: /dev/null missing"; errors=$((errors+1)); }
[ -d /proc ] || { echo "CRITICAL: /proc missing"; errors=$((errors+1)); }

if [ -f /etc/ld.so.preload ]; then
    echo "WARNING: /etc/ld.so.preload is set - this may cause issues"
    errors=$((errors+1))
fi

if [ $errors -eq 0 ]; then
    echo "All critical checks passed - issue may be timing/race related or specific to start-with-nginx.sh"
else
    echo "$errors critical issues found - environment is broken"
fi

echo ""
echo "=== END DEBUG ==="
