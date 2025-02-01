# Beam Damage Analysis
This repository contains a python notebook for analyzing beam damage in a repetition-rate laser system.  It informs the algorithm development for an extended version of the beam mode interlock algorithm used on the high average power ABL beamline at CSU's LAPLEPH.   

## Setup
1. Install Docker and Docker Compose
2. Clone this repository
3. Run `docker-compose up --build`
4. Run the data download script:
   ```bash
   docker-compose exec jupyter python scripts/download_data.py