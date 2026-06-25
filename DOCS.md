# Documentation index

Every Markdown doc in the repo, in one place. Files stay where they are (so
GitHub, Claude Code, and relative links keep working) — this is just the map.

## Start here
| Doc | What it is |
|-----|------------|
| [README.md](README.md) | Project overview + headline results |
| [CLAUDE.md](CLAUDE.md) | Orientation auto-loaded by Claude Code; points to the handoff |
| [HANDOFF.md](HANDOFF.md) | Cross-machine handoff — pending GPU work + exact commands |
| [CHANGELOG.md](CHANGELOG.md) | Inventory of the multi-trajectory evaluation correction |

## Results & methodology
| Doc | What it is |
|-----|------------|
| [Agent_Test/RESULTS_SUMMARY.md](Agent_Test/RESULTS_SUMMARY.md) | Full results writeup: rounds, multi-horizon, direct multi-step, spectral, WPTSE |
| [Agent_Test/REFIT_HANDOFF.md](Agent_Test/REFIT_HANDOFF.md) | How/where to run the pending classical (Ridge/RF/XGBoost) re-fit |

## Modeling plan & agents
| Doc | What it is |
|-----|------------|
| [AI_Agent/MODELING_PLAN.md](AI_Agent/MODELING_PLAN.md) | The R1 → R2 → R3 modeling campaign plan (referenced by `train_all.py`) |
| [.github/agents/wind-cp-forecaster.agent.md](.github/agents/wind-cp-forecaster.agent.md) | Custom Claude Code sub-agent definition |

## Literature
| Doc | What it is |
|-----|------------|
| [Agent_Papers/paper_summaries.md](Agent_Papers/paper_summaries.md) | Canonical literature review (7 papers, Phase 1) — cited by RESULTS_SUMMARY |
| [Agent_Test/paper_summaries.md](Agent_Test/paper_summaries.md) | Earlier forecasting-model literature notes (not referenced elsewhere) |

## Subprojects & infrastructure
| Doc | What it is |
|-----|------------|
| [Agent_Papers/wptse_net/README.md](Agent_Papers/wptse_net/README.md) | WPTSE-Net spectral-synthesis subproject README |
| [Infrastructure/DEPLOY.md](Infrastructure/DEPLOY.md) | Deployment notes (Streamlit / GitHub Pages) |

## Archived (`Agent_Test/Old_Files_Outdated/`)
| Doc | What it is |
|-----|------------|
| [plan-multiAgentWindPressureCp.prompt.md](Agent_Test/Old_Files_Outdated/plan-multiAgentWindPressureCp.prompt.md) | Old multi-agent planning prompt (superseded) |
| [round1_prompt.md](Agent_Test/Old_Files_Outdated/round1_prompt.md) | Round 1 run prompt (Round 1 complete) |

> The thesis lives in a **separate repo** (`../Wind_ML_TimeSeries`); its LaTeX
> sources are under `Thesis/Latex_Document/`.
