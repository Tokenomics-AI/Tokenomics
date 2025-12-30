# Frontend Quick Start Guide

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install flask flask-cors
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment

Make sure you have a `.env` file with your API key:
```
GEMINI_API_KEY=your_api_key_here
```

### 3. Run the Application

```bash
python app.py
```

The server will start on `http://localhost:5000`

### 4. Open in Browser

Navigate to:
- **Home**: http://localhost:5000
- **Dashboard**: http://localhost:5000/dashboard
- **Results**: http://localhost:5000/results

## 📋 Features

### Home Page (`/`)
- Introduction to Tokenomics
- Core components explanation
- Key benefits
- Performance statistics
- Navigation to other pages

### Live Dashboard (`/dashboard`)
- **Run Experiments**: Enter queries and run them in real-time
- **Multiple Modes**:
  - Tokenomics Optimized (full optimization)
  - Baseline (no optimization)
  - A/B Comparison
- **Real-Time Metrics**:
  - Total and average tokens
  - Latency measurements
  - Cache hit rates
- **Interactive Charts**:
  - Token usage per query
  - Latency trends
  - Cache hits vs misses
  - Strategy distribution
- **Detailed Results Table**: See every query's details

### Test Results (`/results`)
- **Aggregated Statistics**: Overall performance metrics
- **Historical Runs**: All past experiments
- **Filtering**: Filter by mode, sort by metrics
- **Detailed Views**: Click any run to see full breakdown

## 🎯 Usage Examples

### Running a Simple Experiment

1. Go to Dashboard
2. Enter query: "What is machine learning?"
3. Select mode: "Tokenomics Optimized"
4. Set queries: 1
5. Click "Run Experiment"
6. View results, charts, and metrics

### Running Multiple Queries

1. Enter query: "Explain Python"
2. Set queries: 5
3. Run experiment
4. See how cache hits improve on repeated queries
5. Observe bandit learning across queries

### Viewing Historical Results

1. Go to Test Results
2. See aggregated statistics
3. Filter by mode if needed
4. Click any run to see details
5. Compare performance across runs

## 🔧 API Endpoints

### Run Experiment
```bash
POST /api/run
Body: {
  "queries": ["your query"],
  "mode": "tokenomics",
  "num_queries": 1
}
```

### Get Status
```bash
GET /api/status
```

### Get All Runs
```bash
GET /api/runs
```

### Get Specific Run
```bash
GET /api/runs/<run_id>
```

### Get Statistics
```bash
GET /api/stats
```

## 📊 Understanding the Dashboard

### Summary Metrics
- **Total Tokens**: Sum of all tokens used
- **Average Tokens**: Average per query
- **Total Latency**: Sum of all response times
- **Average Latency**: Average per query
- **Cache Hit Rate**: Percentage of cache hits
- **Cache Hits**: Number of cache hits

### Charts
1. **Token Usage**: Bar chart showing tokens per query
2. **Latency**: Line chart showing response times
3. **Cache Distribution**: Doughnut chart of hits vs misses
4. **Strategy Distribution**: Pie chart of strategies used

### Results Table
Shows for each query:
- Query number
- Query text
- Tokens used
- Latency
- Cache status (Hit/Miss)
- Strategy selected
- Model used

## 🎨 Customization

### Changing Colors
Edit `static/css/style.css`:
```css
:root {
    --primary-color: #6366f1;  /* Change this */
    --secondary-color: #8b5cf6;
    /* ... */
}
```

### Adding Features
1. **Backend**: Add routes in `app.py`
2. **Frontend**: Add pages in `templates/`
3. **JavaScript**: Add functionality in `static/js/`

## 🐛 Troubleshooting

### Port Already in Use
Change port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### API Errors
- Check `.env` file has correct API key
- Verify API quota hasn't been exceeded
- Check console for error messages

### Charts Not Showing
- Ensure Chart.js is loaded (CDN in base.html)
- Check browser console for JavaScript errors
- Verify data is being returned from API

## 📝 Notes

- Historical runs are saved in `runs_history.json`
- The frontend works even when API quota is exhausted (shows cached data)
- All data is stored locally (no database required)
- The UI is fully responsive and works on mobile

## 🚀 Next Steps

1. Run your first experiment
2. Explore the dashboard features
3. Review historical results
4. Customize for your needs
5. Integrate with your applications

Enjoy optimizing your LLM token usage! 🎉











