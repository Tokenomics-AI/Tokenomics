#!/bin/bash
# Monitor A/B test progress

while true; do
    if [ -f tests/results/ab_test_run.log ]; then
        CURRENT=$(grep -o "\[[0-9]\+/55\]" tests/results/ab_test_run.log | tail -1 | grep -o "[0-9]\+" | head -1)
        if [ -n "$CURRENT" ]; then
            echo "Progress: $CURRENT/55 queries ($(echo "scale=1; $CURRENT*100/55" | bc)%)"
        fi
        
        if grep -q "✅ Test completed" tests/results/ab_test_run.log; then
            echo "Test completed!"
            ls -lh tests/results/AB_COMPARISON_REPORT*.md tests/results/ab_comparison_results*.json tests/results/ab_comparison_results*.csv 2>/dev/null
            break
        fi
    fi
    sleep 30
done








