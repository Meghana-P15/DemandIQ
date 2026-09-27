<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DemandIQ</title>
</head>
<body>

<h1>DemandIQ</h1>
    <h2>Forecast Demand. Detect Risk. Replenish Smarter.</h2>

<h3>Project Overview</h3>
    <p>
        DemandIQ is an end-to-end retail demand intelligence and replenishment
        decision-support system developed from a 12-week machine-learning project.
    </p>

    <p>The system combines:</p>
    <ul>
        <li>Demand forecasting</li>
        <li>High-demand classification</li>
        <li>Behavioural segmentation</li>
        <li>Demand anomaly detection</li>
        <li>Explainable AI</li>
        <li>Replenishment prioritization</li>
        <li>REST API integration</li>
        <li>Web application interface</li>
        <li>Database-backed user and run management</li>
    </ul>

    <h3>Project Evolution</h3>
    <ul>
        <li>Week 1 — Data understanding and EDA</li>
        <li>Week 2 — Data cleaning and feature engineering</li>
        <li>Week 3 — Linear Regression baseline</li>
        <li>Week 4 — Ridge, Lasso and Logistic Regression</li>
        <li>Week 5 — Time-aware cross-validation</li>
        <li>Week 6 — Decision Trees</li>
        <li>Week 7 — Random Forest</li>
        <li>Week 8 — XGBoost + SHAP</li>
        <li>Week 9 — Behavioural segmentation</li>
        <li>Week 10 — Rigorous temporal evaluation</li>
        <li>Week 11 — Demand anomaly detection</li>
        <li>Week 12 — Replenishment decision support</li>
    </ul>

    <h3>Final Model Performance</h3>

    <h4>Demand Regression</h4>
    <ul>
        <li>MAE: 66.1139</li>
        <li>RMSE: 262.4817</li>
        <li>R²: 0.9637</li>
    </ul>

    <h4>High-Demand Classification</h4>
    <ul>
        <li>F1 Score: 0.9618</li>
        <li>ROC-AUC: 0.9976</li>
        <li>PR-AUC: 0.9952</li>
        <li>Selected probability threshold: 0.66</li>
    </ul>

<h3>Replenishment Decision Rule</h3>

    <pre>
0.35 × Demand Score
+ 0.30 × High-Demand Probability
+ 0.15 × Demand Volatility
+ 0.10 × Anomaly Signal
+ 0.10 × Promotion Signal
    </pre>

    <ul>
        <li>Score ≥ 0.75 → CRITICAL</li>
        <li>Score ≥ 0.55 → HIGH</li>
        <li>Score ≥ 0.35 → MEDIUM</li>
        <li>Score &lt; 0.35 → LOW</li>
    </ul>

    <h3>Technology Stack</h3>
    <ul>
        <li>Frontend: Next.js + TypeScript</li>
        <li>Backend: FastAPI + Python</li>
        <li>Machine Learning: scikit-learn + XGBoost + SHAP</li>
        <li>Database: PostgreSQL</li>
        <li>ORM: SQLAlchemy</li>
        <li>Migrations: Alembic</li>
        <li>Testing: pytest</li>
        <li>Deployment: Docker / Docker Compose</li>
    </ul>

    <h3>Important Limitation</h3>
    <p>
        DemandIQ is a decision-support system, not an autonomous inventory
        ordering system. It does not calculate exact order quantities because
        real inventory-on-hand, supplier lead-time and service-level data are
        not available.
    </p>

    <h3>Final Outcome</h3>

    <p>
        DemandIQ transforms historical retail data into demand forecasts,
        high-demand alerts, behavioural insights, anomaly signals and
        replenishment priorities that support smarter retail planning decisions.
    </p>

    <h2>Forecast Demand. Detect Risk. Replenish Smarter.</h2>

</body>
</html>
