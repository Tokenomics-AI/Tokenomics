# Tokenomics Platform - Frontend

## Overview

A modern web frontend for the Tokenomics platform with three main pages:
1. **Home Page** - Introduction and overview
2. **Live Dashboard** - Real-time experimentation and visualization
3. **Test Results** - Historical experiments and evidence

## Features

### Home Page
- Product introduction and narrative
- Core components explanation
- Key benefits showcase
- Performance statistics
- Call-to-action buttons

### Live Dashboard
- Real-time query execution
- Multiple modes (Tokenomics, Baseline, A/B)
- Live metrics visualization
- Interactive charts
- Detailed results table

### Test Results Page
- Historical run listing
- Filtering and sorting
- Aggregated statistics
- Detailed run views
- Performance trends

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (create `.env` file):
```
GEMINI_API_KEY=your_api_key_here
```

3. Run the Flask application:
```bash
python app.py
```

4. Open browser to:
```
http://localhost:5000
```

## Usage

### Running Experiments

1. Navigate to **Live Dashboard**
2. Enter your query
3. Select mode:
   - **Tokenomics Optimized**: Full optimization with caching and bandit
   - **Baseline**: No optimization (for comparison)
   - **A/B Comparison**: Compare both modes
4. Set number of queries
5. Click **Run Experiment**
6. View real-time results, charts, and metrics

### Viewing Results

1. Navigate to **Test Results**
2. View aggregated statistics
3. Filter by mode or sort by metrics
4. Click on any run to see detailed breakdown

## API Endpoints

- `GET /` - Home page
- `GET /dashboard` - Live dashboard
- `GET /results` - Test results page
- `POST /api/run` - Run experiment
- `GET /api/status` - Get platform status
- `GET /api/runs` - Get all runs
- `GET /api/runs/<run_id>` - Get specific run
- `GET /api/stats` - Get aggregated statistics

## Architecture

- **Backend**: Flask REST API
- **Frontend**: HTML/CSS/JavaScript
- **Charts**: Chart.js
- **Styling**: Modern CSS with CSS variables
- **Data Storage**: JSON files (runs_history.json)

## File Structure

```
.
├── app.py                 # Flask backend
├── templates/
│   ├── base.html         # Base template
│   ├── index.html        # Home page
│   ├── dashboard.html    # Live dashboard
│   └── results.html      # Test results
├── static/
│   ├── css/
│   │   └── style.css    # Main stylesheet
│   └── js/
│       ├── main.js      # Utilities and API client
│       ├── dashboard.js  # Dashboard functionality
│       └── results.js    # Results page functionality
└── runs_history.json     # Historical runs data
```

## Features in Detail

### Real-Time Dashboard
- Live query execution
- Real-time metrics updates
- Interactive charts:
  - Token usage per query
  - Latency trends
  - Cache hit/miss distribution
  - Strategy selection distribution
- Detailed results table with all query information

### Historical Results
- Chronological listing of all runs
- Filtering by mode
- Sorting by various metrics
- Detailed run views with:
  - Summary metrics
  - Per-query breakdown
  - Performance analysis

## Development

### Adding New Features

1. **Backend**: Add routes in `app.py`
2. **Frontend**: Add pages in `templates/` and JS in `static/js/`
3. **Styling**: Update `static/css/style.css`

### Testing

1. Run Flask in debug mode:
```bash
python app.py
```

2. Test API endpoints:
```bash
curl http://localhost:5000/api/status
```

## Notes

- The frontend uses mock data when API quota is exhausted
- Historical runs are stored in `runs_history.json`
- Charts are rendered using Chart.js
- The UI is fully responsive

## Future Enhancements

- Real-time WebSocket updates
- Export results to CSV/JSON
- Advanced filtering options
- Performance comparison tools
- User authentication
- Multi-user support











