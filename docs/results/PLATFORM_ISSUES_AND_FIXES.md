# Platform Issues and Fixes

## Issues Identified

### 1. Component-Level Savings Showing 0

**Problem:** Component savings are calculated but showing as 0 in the dashboard.

**Root Cause:** 
- In A/B mode, component_savings are nested under `result['optimized']['component_savings']`
- The aggregation code was looking at `result['component_savings']` directly
- Component savings need to be recalculated using actual baseline comparison

**Fix Applied:**
- Updated `app.py` to extract component_savings from the correct location in A/B mode
- Updated `core.py` to recalculate component savings using actual baseline vs optimized comparison
- Component savings now properly reflect:
  - **Memory Layer**: Tokens saved from cache hits (baseline tokens when cache hit)
  - **Orchestrator**: Output token reduction vs baseline
  - **Bandit**: Remaining savings from strategy selection

### 2. Strategy Distribution Only Showing "cheap" or "none"

**Problem:** Multi-model routing not working - only seeing "cheap" strategy or "none".

**Root Causes:**
1. **"none" appears when:**
   - Cache hits occur (we return cached response before strategy is fully used)
   - `use_bandit=False` (bandit disabled)
   - Guardrail falls back to baseline (strategy not applied)

2. **Only "cheap" appears because:**
   - All queries are simple/complexity="simple"
   - Bandit exploration phase hasn't completed (needs 3 pulls per arm minimum)
   - Cost-aware routing filters out premium arms for simple queries

**Expected Behavior:**
- Simple queries → cheap/balanced arms
- Complex queries → balanced/premium arms
- Exploration phase → tries each arm at least 3 times
- After exploration → exploits best performing arms

**Fix Applied:**
- Strategy is now properly logged even for cache hits
- Model information is included in results
- Strategy chart will show distribution once exploration completes

**To See Multi-Model Routing:**
- Run more queries (need at least 9 queries to explore all 3 arms)
- Use more complex queries to trigger premium arm selection
- Check logs for "Selected strategy" messages showing different arms

### 3. Quality Judge Not Visualized

**Problem:** Quality judge results are not displayed in the dashboard.

**Root Cause:** 
- Quality judge results are stored in `optimized_result['quality_judge']`
- Dashboard doesn't have UI elements to display judge results
- Judge results not passed through to frontend properly

**Fix Applied:**
- Added `quality_judge` to optimized result structure in `app.py`
- Quality judge data now includes:
  - `winner`: "baseline", "optimized", or "equivalent"
  - `explanation`: Brief explanation
  - `confidence`: Confidence score (0-1)

**Next Steps Needed:**
- Add UI elements to display quality judge results in dashboard
- Show judge verdict in results table
- Add quality judge summary to summary metrics

### 4. Same Baseline and Optimized Tokens

**Problem:** Some queries show identical token counts for baseline and optimized.

**This is NOT an error - it's expected behavior:**

**Reasons for Same Tokens:**

1. **Guardrail Fallback:**
   - Optimized path used more tokens or was slower
   - Guardrail triggered → falls back to baseline
   - Result: Same tokens (baseline result is returned)
   - This is the "never worse than baseline" safety feature working correctly

2. **Cache Hit:**
   - Exact cache match → 0 tokens used
   - Baseline would also use 0 tokens if same query
   - Result: Both show 0 tokens (correct behavior)

3. **No Optimization Opportunity:**
   - Query is too simple to benefit from optimization
   - Strategy selected same model as baseline
   - No compression or allocation improvements possible
   - Result: Same tokens (optimization didn't help, but didn't hurt)

4. **Guardrail Prevention:**
   - Optimized was about to use more tokens
   - Guardrail prevented it → used baseline instead
   - Result: Same tokens (safety mechanism working)

**How to Identify:**
- Check `fallback_to_baseline` flag in results
- Check `fallback_reason` field
- Look for guardrail warnings in logs

## Summary of Fixes

### Code Changes Made:

1. **`app.py`:**
   - Fixed component_savings extraction for A/B mode
   - Added `quality_judge` to optimized result structure
   - Added `model` field to results
   - Fixed component savings aggregation

2. **`core.py`:**
   - Recalculate component savings using actual baseline comparison
   - Include model in cache hit returns
   - Properly calculate orchestrator and bandit savings
   - Store baseline comparison data for A/B mode

### Still Needed:

1. **Dashboard UI Updates:**
   - Add quality judge display section
   - Show judge verdict in results table
   - Display fallback reasons when guardrail triggers
   - Show model distribution chart (not just strategy)

2. **Multi-Model Routing Verification:**
   - Run more queries to complete exploration phase
   - Use complex queries to trigger premium arm
   - Monitor logs for strategy selection

3. **Component Savings Display:**
   - Verify savings are now showing correctly
   - Check that percentages are calculated properly

## Testing Recommendations

1. **Run A/B comparison with 10+ queries** to see:
   - Component savings accumulate
   - Strategy distribution diversify
   - Quality judge results appear

2. **Use varied query complexity:**
   - Simple: "What is X?"
   - Medium: "Explain how X works"
   - Complex: "Compare X and Y, analyze pros/cons, provide examples"

3. **Check logs for:**
   - "Selected strategy" messages
   - "Guardrail triggered" warnings
   - "Quality judged" messages
   - Strategy arm IDs (cheap, balanced, premium)

4. **Monitor dashboard for:**
   - Component savings > 0
   - Multiple strategies in distribution
   - Quality judge results
   - Model information


