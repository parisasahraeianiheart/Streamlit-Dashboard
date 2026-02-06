# Streamlit-Dashboard

Project Summary

This project analyzes the relationship between time on page (user engagement) and revenue using session-level data. The goal was to determine whether spending more time on a page is associated with higher revenue and how that relationship changes after accounting for other factors.

What Was Done
	•	Cleaned and prepared raw data (handled missing values, type conversions, and profiling)
	•	Performed exploratory data analysis with visualizations and correlation metrics
	•	Built two models:
	•	Simple model: revenue ~ time on page
	•	Controlled model: revenue ~ time on page + other variables
	•	Compared results to evaluate the impact of contextual factors
	•	Developed a Streamlit app and automated PDF report for non-technical audiences

Key Findings
	•	The raw data showed a negative association between time on page and revenue.
	•	After controlling for variables such as platform and site context, the relationship became positive.
	•	This indicates that time on page is meaningful, but only when interpreted within the right context.

Business Takeaways
	•	Engagement metrics should not be used in isolation
	•	Context matters when evaluating user behavior
	•	Improving meaningful engagement can support higher revenue

Tools Used

Python (pandas, matplotlib, statsmodels), Streamlit, automated reporting
