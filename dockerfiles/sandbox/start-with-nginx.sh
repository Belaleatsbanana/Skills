#!/bin/bash
# Start nginx load balancer with multiple uwsgi workers
# Supports both single-node (unix sockets) and multi-node Slurm (TCP) modes

set -e

export NUM_WORKERS=${NUM_WORKERS:-$(nproc --all)}

# =============================================================================
# Multi-node mode detection
# =============================================================================
# Enable multi-node mode when:
#   1. NEMO_SKILLS_SANDBOX_MULTINODE=1 is explicitly set, OR
#   2. Running under Slurm with SLURM_NNODES > 1
MULTINODE_MODE=0
if [ "${NEMO_SKILLS_SANDBOX_MULTINODE:-0}" = "1" ]; then
    MULTINODE_MODE=1
    echo "Multi-node mode enabled via NEMO_SKILLS_SANDBOX_MULTINODE=1"
elif [ -n "$SLURM_JOB_NODELIST" ] && [ "${SLURM_NNODES:-1}" -gt 1 ]; then
    MULTINODE_MODE=1
    echo "Multi-node mode auto-detected: SLURM_NNODES=${SLURM_NNODES}"
fi

# Determine master node and current node identity
if [ "$MULTINODE_MODE" = "1" ]; then
    # SLURM_ALL_HOSTNAMES is set by nemo-run before container starts
    if [ -n "$SLURM_ALL_HOSTNAMES" ]; then
        echo "Using SLURM_ALL_HOSTNAMES: $SLURM_ALL_HOSTNAMES"
        ALL_NODES="$SLURM_ALL_HOSTNAMES"
        MASTER_NODE="${SLURM_MASTER_NODE:-$(echo "$ALL_NODES" | awk '{print $1}')}"
        NODE_COUNT=$(echo "$ALL_NODES" | wc -w)
    else
        echo "ERROR: SLURM_ALL_HOSTNAMES not set. This should be set by nemo-run."
        echo "Falling back to single node mode."
        MASTER_NODE="${SLURM_MASTER_NODE:-$(hostname)}"
        ALL_NODES="$MASTER_NODE"
        NODE_COUNT=1
    fi

    CURRENT_NODE=$(hostname)
    # Normalize hostnames (strip domain if present for comparison)
    CURRENT_NODE_SHORT="${CURRENT_NODE%%.*}"
    MASTER_NODE_SHORT="${MASTER_NODE%%.*}"

    if [ "$CURRENT_NODE_SHORT" = "$MASTER_NODE_SHORT" ]; then
        IS_MASTER=1
        echo "This node ($CURRENT_NODE) is the MASTER node"
    else
        IS_MASTER=0
        echo "This node ($CURRENT_NODE) is a WORKER node (master: $MASTER_NODE)"
    fi

    # TCP mode: workers listen on ports instead of unix sockets
    SANDBOX_WORKER_BASE_PORT=${SANDBOX_WORKER_BASE_PORT:-$((NGINX_PORT + 1))}
    echo "Multi-node configuration:"
    echo "  Master node: $MASTER_NODE"
    echo "  Total nodes: $NODE_COUNT"
    echo "  Workers per node: $NUM_WORKERS"
    echo "  Worker port range: $SANDBOX_WORKER_BASE_PORT to $((SANDBOX_WORKER_BASE_PORT + NUM_WORKERS - 1))"
else
    IS_MASTER=1  # In single-node mode, we're always the master
    echo "Starting single-node deployment with nginx (unix sockets upstream)..."
fi

echo "Workers per node: $NUM_WORKERS, Nginx port: $NGINX_PORT"

# =============================================================================
# uWSGI configuration
# =============================================================================
# Allow callers to opt-out of single-process state-preserving mode where each worker is given one process
: "${STATEFUL_SANDBOX:=1}"
if [ "$STATEFUL_SANDBOX" -eq 1 ]; then
    UWSGI_PROCESSES=1
    UWSGI_CHEAPER=1
else
    # In stateless mode, honour caller-supplied values
    : "${UWSGI_PROCESSES:=1}"
    : "${UWSGI_CHEAPER:=1}"
fi

export UWSGI_PROCESSES UWSGI_CHEAPER

echo "UWSGI settings: PROCESSES=$UWSGI_PROCESSES, CHEAPER=$UWSGI_CHEAPER"

# Validate and fix uwsgi configuration
if [ -z "$UWSGI_PROCESSES" ]; then
    UWSGI_PROCESSES=2
fi

if [ -z "$UWSGI_CHEAPER" ]; then
    UWSGI_CHEAPER=1
elif [ "$UWSGI_CHEAPER" -le 0 ]; then
    echo "WARNING: UWSGI_CHEAPER ($UWSGI_CHEAPER) must be at least 1"
    UWSGI_CHEAPER=1
    echo "Setting UWSGI_CHEAPER to $UWSGI_CHEAPER"
elif [ "$UWSGI_CHEAPER" -ge "$UWSGI_PROCESSES" ]; then
    echo "WARNING: UWSGI_CHEAPER ($UWSGI_CHEAPER) must be lower than UWSGI_PROCESSES ($UWSGI_PROCESSES)"
    if [ "$UWSGI_PROCESSES" -eq 1 ]; then
        # For single process, disable cheaper mode entirely
        echo "Disabling cheaper mode for single process setup"
        UWSGI_CHEAPER=""
    else
        UWSGI_CHEAPER=$((UWSGI_PROCESSES - 1))
        echo "Setting UWSGI_CHEAPER to $UWSGI_CHEAPER"
    fi
fi

export UWSGI_PROCESSES
if [ -n "$UWSGI_CHEAPER" ]; then
    export UWSGI_CHEAPER
    echo "UWSGI config - Processes: $UWSGI_PROCESSES, Cheaper: $UWSGI_CHEAPER"
else
    echo "UWSGI config - Processes: $UWSGI_PROCESSES, Cheaper: disabled"
fi

# =============================================================================
# Generate nginx upstream configuration
# =============================================================================
echo "Generating nginx configuration..."

UPSTREAM_FILE="/tmp/upstream_servers.conf"
> $UPSTREAM_FILE  # Clear the file

if [ "$MULTINODE_MODE" = "1" ]; then
    # Multi-node: generate TCP upstream entries for all nodes × all worker ports
    echo "Generating upstream servers for multi-node (TCP)..."
    for node in $ALL_NODES; do
        for i in $(seq 1 $NUM_WORKERS); do
            WORKER_PORT=$((SANDBOX_WORKER_BASE_PORT + i - 1))
            echo "        server ${node}:${WORKER_PORT} max_fails=3 fail_timeout=30s;" >> $UPSTREAM_FILE
        done
    done
    echo "Generated upstream servers for $NODE_COUNT nodes × $NUM_WORKERS workers (TCP):"
else
    # Single-node: use unix sockets (original behavior)
    SOCKET_DIR="/tmp/uwsgi_sockets"
    mkdir -p "$SOCKET_DIR"
    chmod 777 "$SOCKET_DIR"

    for i in $(seq 1 $NUM_WORKERS); do
        SOCKET_PATH="${SOCKET_DIR}/worker${i}.sock"
        echo "        server unix:${SOCKET_PATH} max_fails=3 fail_timeout=30s;" >> $UPSTREAM_FILE
    done
    echo "Generated upstream servers for $NUM_WORKERS workers (unix sockets):"
fi

cat $UPSTREAM_FILE

# Only the master node generates and uses nginx config
if [ "$IS_MASTER" = "1" ]; then
    # Create nginx config by replacing placeholders
    sed "s|\${NGINX_PORT}|${NGINX_PORT}|g" /etc/nginx/nginx.conf.template > /tmp/nginx_temp.conf

    # Replace the upstream servers placeholder with the actual servers
    awk -v upstream_file="$UPSTREAM_FILE" '
    /\${UPSTREAM_SERVERS}/ {
        while ((getline line < upstream_file) > 0) {
            print line
        }
        close(upstream_file)
        next
    }
    { print }
    ' /tmp/nginx_temp.conf > /etc/nginx/nginx.conf

    echo "Nginx configuration created successfully"

    # Test nginx configuration
    echo "Testing nginx configuration..."
    if ! nginx -t; then
        echo "ERROR: nginx configuration test failed"
        echo "Generated nginx.conf:"
        cat /etc/nginx/nginx.conf
        exit 1
    fi
fi

# =============================================================================
# Log setup
# =============================================================================
mkdir -p /var/log/nginx
# Remove symlinks if present and create real log files
rm -f /var/log/nginx/access.log /var/log/nginx/error.log
touch /var/log/nginx/access.log /var/log/nginx/error.log
chmod 644 /var/log/nginx/*.log
# Pre-create per-worker log files so uWSGI writes to regular files
for i in $(seq 1 $NUM_WORKERS); do
    touch /var/log/worker${i}.log
done
chmod 644 /var/log/worker*.log || true

# Mirror logs to stdout/stderr for docker logs
tail -f /var/log/nginx/access.log &> /dev/stdout &
tail -f /var/log/nginx/error.log &> /dev/stderr &
tail -f /var/log/worker*.log &> /dev/stderr &

# =============================================================================
# Worker management
# =============================================================================
echo "Starting $NUM_WORKERS workers in parallel..."
WORKER_PIDS=()

# Function to cleanup on exit
cleanup() {
    echo "Shutting down workers and nginx..."
    for pid in "${WORKER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    pkill -f nginx || true
    # Cleanup leftover unix sockets (single-node mode only)
    if [ "$MULTINODE_MODE" != "1" ] && [ -n "$SOCKET_DIR" ] && [ -d "$SOCKET_DIR" ]; then
        rm -f "$SOCKET_DIR"/worker*.sock 2>/dev/null || true
    fi
    exit 0
}

trap cleanup SIGTERM SIGINT

# Function to start a single worker and return its PID
start_worker() {
    local i=$1

    if [ "$MULTINODE_MODE" = "1" ]; then
        # Multi-node: use TCP socket
        local WORKER_PORT=$((SANDBOX_WORKER_BASE_PORT + i - 1))
        local SOCKET_SPEC="0.0.0.0:${WORKER_PORT}"
        local SOCKET_DISPLAY="${WORKER_PORT}"
        echo "Starting worker $i on TCP port $WORKER_PORT..." >&2
    else
        # Single-node: use unix socket
        local SOCKET_PATH="${SOCKET_DIR}/worker${i}.sock"
        local SOCKET_SPEC="${SOCKET_PATH}"
        local SOCKET_DISPLAY="${SOCKET_PATH}"

        # Ensure old socket is removed if present
        if [ -S "$SOCKET_PATH" ]; then
            rm -f "$SOCKET_PATH"
        fi
        echo "Starting worker $i on socket $SOCKET_PATH..." >&2
    fi

    # Create a custom uwsgi.ini for this worker
    cat > /tmp/worker${i}_uwsgi.ini << EOF
[uwsgi]
module = main
callable = app
processes = ${UWSGI_PROCESSES}
http-socket = ${SOCKET_SPEC}
vacuum = true
master = true
die-on-term = true
memory-report = true

# Connection and request limits to prevent overload
listen = 100
http-timeout = 300
socket-timeout = 300

# NO auto-restart settings to preserve session persistence
# max-requests and reload-on-rss would kill Jupyter kernels

# Logging for debugging 502 errors
disable-logging = false
log-date = true
log-prefix = [worker${i}]
logto = /var/log/worker${i}.log
EOF

    # Add chmod-socket only for unix sockets
    if [ "$MULTINODE_MODE" != "1" ]; then
        echo "chmod-socket = 666" >> /tmp/worker${i}_uwsgi.ini
    fi

    if [ -n "$UWSGI_CHEAPER" ]; then
        echo "cheaper = ${UWSGI_CHEAPER}" >> /tmp/worker${i}_uwsgi.ini
    fi

    echo "Created custom uwsgi config for worker $i (${SOCKET_DISPLAY})" >&2

    # Start worker with custom config
    (
        # Run uwsgi from /app in a subshell so the current directory of the main script is unaffected
        cd /app && env WORKER_NUM=$i uwsgi --ini /tmp/worker${i}_uwsgi.ini
    ) &

    local pid=$!
    echo "Worker $i started with PID $pid on ${SOCKET_DISPLAY}" >&2
    echo $pid
}

# Start all workers simultaneously
for i in $(seq 1 $NUM_WORKERS); do
    pid=$(start_worker $i)
    WORKER_PIDS+=($pid)
done

echo "All $NUM_WORKERS workers started simultaneously - waiting for readiness..."

# =============================================================================
# Wait for workers to be ready
# =============================================================================
echo "Waiting for workers to start..."
READY_WORKERS=0
TIMEOUT=180  # Increased timeout since uwsgi takes time to start
START_TIME=$(date +%s)

# Track which workers are ready to avoid redundant checks
declare -A WORKER_READY

while [ $READY_WORKERS -lt $NUM_WORKERS ]; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    if [ $ELAPSED -gt $TIMEOUT ]; then
        echo "ERROR: Timeout waiting for workers to start"

        # Show worker status and logs
        echo "Worker status:"
        for i in "${!WORKER_PIDS[@]}"; do
            pid=${WORKER_PIDS[$i]}
            worker_num=$((i+1))
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Worker $worker_num (PID $pid): Process Running"
                echo "    Recent log output:"
                tail -20 /var/log/worker${worker_num}.log 2>/dev/null | sed 's/^/      /' || echo "      No log found"
            else
                echo "  Worker $worker_num (PID $pid): Dead"
                echo "    Log:"
                tail -30 /var/log/worker${worker_num}.log 2>/dev/null | sed 's/^/      /' || echo "      No log found"
            fi
        done

        exit 1
    fi

    READY_WORKERS=0
    for i in $(seq 1 $NUM_WORKERS); do
        # Skip workers that are already ready
        if [ "${WORKER_READY[$i]}" = "1" ]; then
            READY_WORKERS=$((READY_WORKERS + 1))
            continue
        fi

        # Try the health check
        if [ "$MULTINODE_MODE" = "1" ]; then
            WORKER_PORT=$((SANDBOX_WORKER_BASE_PORT + i - 1))
            HEALTH_URL="http://127.0.0.1:${WORKER_PORT}/health"
            if curl -s -f --connect-timeout 2 --max-time 5 "$HEALTH_URL" > /dev/null 2>&1; then
                READY_WORKERS=$((READY_WORKERS + 1))
                WORKER_READY[$i]=1
                echo "  Worker $i (port $WORKER_PORT): Ready! ($READY_WORKERS/$NUM_WORKERS)"
            fi
        else
            SOCKET_PATH="${SOCKET_DIR}/worker${i}.sock"
            if curl -s -f --connect-timeout 2 --max-time 5 --unix-socket "$SOCKET_PATH" http://localhost/health > /dev/null 2>&1; then
                READY_WORKERS=$((READY_WORKERS + 1))
                WORKER_READY[$i]=1
                echo "  Worker $i (socket $SOCKET_PATH): Ready! ($READY_WORKERS/$NUM_WORKERS)"
            fi
        fi
    done

    # Show progress every 10 seconds
    if [ $((ELAPSED % 10)) -eq 0 ] && [ $READY_WORKERS -lt $NUM_WORKERS ]; then
        echo "  Progress: $READY_WORKERS/$NUM_WORKERS workers ready (${ELAPSED}s elapsed)"
    fi

    # Check less frequently to reduce CPU usage and log spam
    sleep 2
done

echo "All local workers are ready!"

# =============================================================================
# Start nginx (master node only)
# =============================================================================
if [ "$IS_MASTER" = "1" ]; then
    if [ "$MULTINODE_MODE" = "1" ]; then
        # In multi-node mode, optionally wait for remote workers before starting nginx
        WAIT_FOR_REMOTE=${NEMO_SKILLS_SANDBOX_WAIT_FOR_REMOTE:-0}
        if [ "$WAIT_FOR_REMOTE" = "1" ] && [ "$NODE_COUNT" -gt 1 ]; then
            echo "Waiting for remote workers to be ready..."
            REMOTE_READY=0
            REMOTE_TIMEOUT=300
            REMOTE_START=$(date +%s)

            while true; do
                REMOTE_ELAPSED=$(($(date +%s) - REMOTE_START))
                if [ $REMOTE_ELAPSED -gt $REMOTE_TIMEOUT ]; then
                    echo "WARNING: Timeout waiting for all remote workers, starting nginx anyway"
                    break
                fi

                TOTAL_READY=0
                TOTAL_EXPECTED=$((NODE_COUNT * NUM_WORKERS))
                for node in $ALL_NODES; do
                    for i in $(seq 1 $NUM_WORKERS); do
                        WORKER_PORT=$((SANDBOX_WORKER_BASE_PORT + i - 1))
                        if curl -s -f --connect-timeout 1 --max-time 2 "http://${node}:${WORKER_PORT}/health" > /dev/null 2>&1; then
                            TOTAL_READY=$((TOTAL_READY + 1))
                        fi
                    done
                done

                if [ $TOTAL_READY -ge $TOTAL_EXPECTED ]; then
                    echo "All $TOTAL_READY/$TOTAL_EXPECTED remote workers ready!"
                    break
                fi

                if [ $((REMOTE_ELAPSED % 10)) -eq 0 ]; then
                    echo "  Remote progress: $TOTAL_READY/$TOTAL_EXPECTED workers ready (${REMOTE_ELAPSED}s elapsed)"
                fi
                sleep 2
            done
        fi
    fi

    echo "Starting nginx on port $NGINX_PORT..."
    nginx

    # Enable network blocking for user code execution if requested
    # This MUST happen AFTER nginx/uwsgi start (they need sockets for API)
    # Using /etc/ld.so.preload ensures this cannot be bypassed by user code
    BLOCK_NETWORK_LIB="/usr/lib/libblock_network.so"
    if [ "${NEMO_SKILLS_SANDBOX_BLOCK_NETWORK:-0}" = "1" ]; then
        if [ -f "$BLOCK_NETWORK_LIB" ]; then
            echo "$BLOCK_NETWORK_LIB" > /etc/ld.so.preload
            echo "Network blocking ENABLED: All new processes will have network blocked"
            echo "  (API server sockets created before this, so API still works)"
        else
            echo "WARNING: Network blocking requested but $BLOCK_NETWORK_LIB not found"
        fi
    fi
else
    # Worker node in multi-node mode: start a local nginx proxy that forwards to master
    # This allows clients to connect to localhost:NGINX_PORT on any node
    echo "Starting local nginx proxy to master node..."

    # Generate a simple proxy config for worker nodes
    cat > /etc/nginx/nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    # Proxy all requests to the master node's nginx LB
    upstream master_lb {
        server ${MASTER_NODE}:${NGINX_PORT};
    }

    server {
        listen ${NGINX_PORT};
        server_name localhost;

        client_max_body_size 10M;
        client_body_buffer_size 128k;

        location / {
            proxy_pass http://master_lb;

            # Forward all headers including X-Session-ID for consistent hashing
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
            proxy_set_header X-Session-ID \$http_x_session_id;

            # Match master timeouts
            proxy_connect_timeout 1200s;
            proxy_send_timeout 1200s;
            proxy_read_timeout 1200s;

            proxy_buffering off;
        }

        location /nginx-status {
            stub_status on;
            access_log off;
            allow 127.0.0.1;
            allow ::1;
            deny all;
        }
    }

    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log warn;
}
EOF

    echo "Testing nginx proxy configuration..."
    if ! nginx -t; then
        echo "ERROR: nginx proxy configuration test failed"
        cat /etc/nginx/nginx.conf
        exit 1
    fi

    nginx
    echo "Local nginx proxy started on port $NGINX_PORT -> master $MASTER_NODE:$NGINX_PORT"
fi

# =============================================================================
# Print status summary
# =============================================================================
if [ "$IS_MASTER" = "1" ]; then
    echo "=== Sandbox deployment ready (MASTER) ==="
    echo "Nginx load balancer: http://localhost:$NGINX_PORT"
    echo "Session affinity: enabled (based on X-Session-ID header)"
    if [ "$MULTINODE_MODE" = "1" ]; then
        echo "Mode: MULTI-NODE (TCP)"
        echo "Nodes: $NODE_COUNT ($ALL_NODES)"
        echo "Workers per node: $NUM_WORKERS"
        echo "Total workers: $((NODE_COUNT * NUM_WORKERS))"
        echo "Worker port range: $SANDBOX_WORKER_BASE_PORT to $((SANDBOX_WORKER_BASE_PORT + NUM_WORKERS - 1))"
    else
        echo "Mode: SINGLE-NODE (unix sockets)"
        echo "Workers: $NUM_WORKERS (unix sockets in ${SOCKET_DIR}/worker{1..$NUM_WORKERS}.sock)"
    fi
    echo "Nginx status: http://localhost:$NGINX_PORT/nginx-status"
else
    echo "=== Sandbox deployment ready (WORKER NODE) ==="
    echo "Mode: MULTI-NODE (TCP) - worker node"
    echo "Local nginx proxy: http://localhost:$NGINX_PORT -> master $MASTER_NODE:$NGINX_PORT"
    echo "Master node: $MASTER_NODE (nginx LB with consistent hash routing)"
    echo "Local workers: $NUM_WORKERS on ports $SANDBOX_WORKER_BASE_PORT to $((SANDBOX_WORKER_BASE_PORT + NUM_WORKERS - 1))"
fi

echo "UWSGI processes per worker: $UWSGI_PROCESSES"
if [ -n "$UWSGI_CHEAPER" ]; then
    echo "UWSGI cheaper mode: $UWSGI_CHEAPER"
else
    echo "UWSGI cheaper mode: disabled"
fi

# Show process status
echo "Process status:"
for i in "${!WORKER_PIDS[@]}"; do
    pid=${WORKER_PIDS[$i]}
    if kill -0 "$pid" 2>/dev/null; then
        echo "  Worker $((i+1)) (PID $pid): Running"
    else
        echo "  Worker $((i+1)) (PID $pid): Dead"
    fi
done

# =============================================================================
# Monitoring loop
# =============================================================================
echo "Monitoring processes (Ctrl+C to stop)..."

# Only run load monitor on master node
if [ "$IS_MASTER" = "1" ]; then
    monitor_load() {
        echo "Starting worker load monitor (updates every 60s)..."
        while true; do
            sleep 60
            echo "--- Worker Load Stats (Top 10) at $(date) ---"
            grep "upstream:" /var/log/nginx/access.log | awk -F'upstream: ' '{print $2}' | awk -F' session: ' '{print $1}' | sort | uniq -c | sort -nr | head -n 10 || echo "No logs yet"
            echo "--- End Stats ---"
        done
    }
    monitor_load &  # Run in background
fi

while true; do
    # Check if any worker died
    for idx in "${!WORKER_PIDS[@]}"; do
        pid=${WORKER_PIDS[$idx]}
        i=$((idx + 1))
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "WARNING: Worker $i (PID $pid) died - restarting..."
            new_pid=$(start_worker $i)
            WORKER_PIDS[$idx]=$new_pid
        fi
    done

    # Check nginx (runs on all nodes in multi-node mode)
    if [ "$IS_MASTER" = "1" ] || [ "$MULTINODE_MODE" = "1" ]; then
        if ! pgrep nginx > /dev/null; then
            echo "ERROR: Nginx died unexpectedly"
            cleanup
            exit 1
        fi
    fi

    sleep 10
done
