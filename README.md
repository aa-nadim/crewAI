# Crew-AI

```bash

python3 -m venv .venv
source .venv/bin/activate

source .venv/Scripts/activate # for windows


pip install --upgrade pip
pip install -r requirements.txt
```

## Muti AI Agent Run

```bash
# 001
python multi-ai-agent/001_create-agents-to-research-and-write-an-article/research-and-write.py

#002
python multi-ai-agent/002_multi-agent-customer-support-automation/customer-support.py

#003
python multi-ai-agent/003_tools-for-a-customer-outreach-campaign/customer-outreach-campaign.py

#004
cd multi-ai-agent/004_automate-event-planning
python automate-event-planning.py

#005
python multi-ai-agent/005_mutli-agent-collaboration-for-financial-analysis/financial-analysis.py

#006
cd multi-ai-agent/006_build-a-crew-to-trailor-job-applications/
python job-applications.py

```

## Practical Multi AI Agents Run
```bash

#001
cd practical-multi-ai-agents/001_automated-project_planning-estimation-and-allocation/
python planning-estimation-allocation.py

#002
cd practical-multi-ai-agents/002_building-project-progress-report/
python project-progress-report.py


#003
cd practical-multi-ai-agents/003_agentic-sales-pipeline
python agentic-sales-pipeline.py

#004
cd practical-multi-ai-agents/004_support-data-insight-analysis
python support-data-insight-analysis.py

#005
cd practical-multi-ai-agents/005_content-creation-at-scale
python content-creation-at-scale.py


#006
cd practical-multi-ai-agents/006_blog-post-crew-in-production

deactivate
python3 -m venv .venv
source .venv/bin/activate
source .venv/Scripts/activate # for windows

pip install crewai
crewai create crew new_project

cd new_project

crewai install
crewai run

``