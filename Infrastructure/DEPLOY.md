# Infrastructure — Deployment Guide

This folder documents and mirrors all deployment-related files for the
Wind Pressure Cp Forecasting project.

> ⚠️ **Important**: The *live* files that GitHub and Streamlit actually read
> must stay at specific locations in the repo root. This folder contains
> **reference copies** + the deployment instructions.

---

## Architecture

```
Internet user
     │
     ▼
┌─────────────────────────────────┐
│  GitHub Pages (static site)     │  cesar421.github.io/Wind_BWU/
│  docs/index.html  ◄── root      │  ← project overview, results, links
└────────────────┬────────────────┘
                 │  "Launch Dashboard" button
                 ▼
┌─────────────────────────────────┐
│  Streamlit Cloud (dynamic app)  │  *.streamlit.app
│  AI_Agent/streamlit_app.py      │  ← interactive plots, model inference
└─────────────────────────────────┘
```

---

## 1. GitHub Pages

### Live files (must stay at root)
| File | Location in repo | Purpose |
|------|-----------------|---------|
| `index.html` | `docs/index.html` | Main webpage |

### Activation steps (one-time)
1. Push the `docs/` folder to GitHub
2. Go to **github.com/Cesar421/Wind_BWU → Settings → Pages**
3. Source: `Deploy from a branch`
4. Branch: `main` · Folder: `/docs`
5. Click **Save**
6. URL will be: `https://cesar421.github.io/Wind_BWU/`

### Update after deployment
Once you have the Streamlit URL, replace `STREAMLIT_URL` in `docs/index.html`
(3 occurrences: lines ~175, ~249, ~272).

---

## 2. Streamlit Cloud

### Live files (must stay at root or specific paths)
| File | Location in repo | Purpose |
|------|-----------------|---------|
| `requirements.txt` | `/requirements.txt` (root) | Python dependencies |
| `streamlit_app.py` | `AI_Agent/streamlit_app.py` | Main app entry point |

### Deploy steps (one-time, ~5 minutes)
1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with your GitHub account (Cesar421)
3. Click **"New app"**
4. Fill in:
   - Repository: `Cesar421/Wind_BWU`
   - Branch: `main`
   - Main file path: `AI_Agent/streamlit_app.py`
5. Click **Deploy**
6. You'll get a URL like: `https://cesar421-wind-bwu-xxxx.streamlit.app`

### Update the GitHub Pages link
After getting the Streamlit URL, run in the terminal:
```powershell
# Replace STREAMLIT_URL with your actual URL in docs/index.html and README.md
# Then commit and push:
git add docs/index.html README.md
git commit -m "fix: add Streamlit app URL to GitHub Pages"
git push
```

---

## 3. File Map

```
Wind_BWU/                          ← repo root
├── docs/                          ← GitHub Pages (MUST be here)
│   └── index.html                 ← project website
├── requirements.txt               ← Streamlit Cloud deps (MUST be here)
├── AI_Agent/
│   └── streamlit_app.py           ← Streamlit entry point
│
└── Infrastructure/                ← THIS FOLDER (documentation + copies)
    ├── DEPLOY.md                  ← this file
    ├── github_pages/
    │   └── index.html             ← reference copy of docs/index.html
    └── streamlit/
        ├── requirements.txt       ← reference copy of root requirements.txt
        └── streamlit_app_reference.py  ← reference copy of the app
```

---

## 4. Updating the page

After any change to `docs/index.html`:
```powershell
git add docs/index.html Infrastructure/github_pages/index.html
git commit -m "docs: update GitHub Pages"
git push
```
GitHub Pages redeploys automatically in ~60 seconds.

After any change to `streamlit_app.py`:
```powershell
git add AI_Agent/streamlit_app.py Infrastructure/streamlit/streamlit_app_reference.py
git commit -m "feat: update Streamlit dashboard"
git push
```
Streamlit Cloud redeploys automatically on push.
