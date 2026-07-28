# Daily MLB Picks — GitHub Actions Workflow

This folder contains the workflow file to add to the **[jupitermoon7/Ai-assistant](https://github.com/jupitermoon7/Ai-assistant)** repo so it automatically posts fresh MLB picks to the Agent Predictions page each morning.

---

## Setup (one-time)

### 1. Copy the workflow file

In the **Ai-assistant** repo, create the file at this exact path:

```
.github/workflows/daily-picks.yml
```

Copy the contents of `daily-picks.yml` from this folder verbatim.

### 2. Add GitHub Secrets

In the **Ai-assistant** repo → **Settings → Secrets and variables → Actions**, add these three secrets:

| Secret name | Value |
|---|---|
| `MLB_APP_URL` | Your Replit app URL, e.g. `https://6dc55238-791d-4969-a099-33d6f9ead5c4-00-2j7etasxo8fyx-usvmuwmm.kirk.replit.dev` |
| `AGENT_API_TOKEN` | The `AGENT_API_TOKEN` value from this Replit project's Secrets panel |
| `OPENAI_API_KEY` | An OpenAI API key with GPT-4o access |

> **Tip:** If you've deployed the app to a stable URL (Replit Deployments), use that URL instead of the `.replit.dev` dev domain.

### 3. Trigger a test run

After adding the file and secrets, go to **Actions → Daily MLB Picks → Run workflow** and click the green button. Check the run log — it should print something like:

```
Fetching agent data…
  plays=18, hr_threats=22, locks=5, nrfi=6, f5=9
Running GPT-4o analysis…
  got 5 picks + 2 fades
Posting picks to MLB Lab…
  stored: id=..., picks_count=7
Done ✓
```

### 4. Scheduled runs

The workflow runs automatically every day at **11:00 AM Eastern Time** (16:00 UTC). Picks appear in the **External Agent Picks** section on the Agent Predictions page (`/agent-predictions`) within a minute of the run completing.

---

## How it works

1. **Fetch** — `GET /api/agent-data` pulls tonight's plays, HR threats, NRFI, locks, and F5 leans from the MLB Lab model.
2. **Analyse** — GPT-4o compares model projections vs live sportsbook lines and picks the top 5 plays with the biggest market edge, plus 1-2 fades.
3. **Submit** — `POST /api/agent-predictions/submit` stores the picks in the MLB Lab database tagged as `Ai-assistant (GitHub)`.
4. **Display** — The Agent Predictions page loads them automatically on every refresh.
