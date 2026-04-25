#!/bin/bash
cd /vercel/share/v0-project

# Stage all changes
git add -A

# Commit with a descriptive message
git commit -m "feat: Add comprehensive EchoChamber simulation interface

- Add vis.js network graph visualization for 500 agents
- Add Vega-Lite charts for polarization index and echo chamber coefficient  
- Add belief distribution histogram with dynamic updates
- Add control panel with alpha/beta/epsilon sliders
- Add start/pause/reset simulation controls
- Add scenario presets (Chronological, Engagement, Diversity)
- Add agent inspector panel for detailed node info
- Add simulation store with SSE streaming integration
- Update styling with dark theme and responsive layout

Co-authored-by: v0[bot] <v0[bot]@users.noreply.github.com>"

# Push to current branch
git push origin HEAD
