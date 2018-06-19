# An analysis of GitHub data

I used a subset of GitHub data to run a short exploratory analysis and create an information retrieval system that uses the project description to return relevant results.

The data come from [GHTorrent](http://ghtorrent.org/) that was accessed through **Google BigQuery**.

A walkthrough of the analysis is available through the notebooks:

* **001-eda-github-data.ipynb**: Simple exploration of the GitHub data subset.
* **002-spatial-eda-example.ipynb**: Visualised the spatial distribution of user registrations in Germany and UK.
* **003-text-preprocessing-and-modelling.ipynb**: Preprocessed GitHub project descriptions and trained word and paragraph vectors.
* **004-information-retrieval-and-data-visualisation.ipynb**: Created an IR system based on word embeddings and visualised some of the queries. I mainly searched for projects on **Machine Learning**, **Virtual Reality**, **Robotics** and **Blockchain**.
* **005-correlation-analysis.ipynb**: Examined how three variables, the number of user accounts, projects and their spatial diversity, are correlated with the EIS indicators.

**Note**: The `/models` and `/data` directories are not included due to their size. Contact me if you want to reproduce the work or use the data in your project.