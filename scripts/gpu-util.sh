#!/bin/bash
# Print GFX utilization % of the first GPU (0 if unparseable).
/opt/rocm/bin/amd-smi 2>/dev/null | python3 -c "
import sys, re
for line in sys.stdin:
    if 'N/A' in line and '%' in line:
        m = re.findall(r'(\d+)\s*%', line)
        if m:
            print(m[0])
            break
else:
    print(0)
"
