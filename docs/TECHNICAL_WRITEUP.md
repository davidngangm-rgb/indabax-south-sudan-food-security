# Technical write-up

I developed this solution using only the organizer-provided competition data. The binary target is whether a county-period is in Crisis-or-worse food insecurity (IPC Phase 3+). My selected final models are State Horizon and Interval Trend, two deterministic `HistGradientBoostingClassifier` pipelines.

## A. Features used

I use the provided state, county, population, period year/month, preceding IPC phase, preceding Phase-3+-population percentage, and prior-year cereal production and gap variables. I derive numeric IPC indicators, IPC crisis and emergency flags, production/gap per-capita and gap-to-production ratios, seasonal sine/cosine terms, absolute cereal gap, and interactions between preceding IPC, Phase-3+ percentage, and cereal gap.

For both final models I construct county-period history: strictly prior lags, expanding historical means, recent rolling means, changes, accelerations, and recent-versus-history differences. State Horizon additionally uses strictly prior state-level aggregates. Interval Trend instead includes elapsed-time normalized historical changes and prior assessment intervals.

## B. Features excluded and why

I do not use target-derived history in either final model. I also do not use current-period IPC information, test labels, external datasets, AutoML, or leaderboard feedback. Same-period observations are aggregated only before being shifted for later periods; they are never used as each other's history. These exclusions prevent target and temporal leakage.

## C. Validation approach

I use chronological forward validation for 2023-09, 2023-12, 2024-04, 2024-09, 2024-12, and 2025-04. Each validation period is fitted only with rows from strictly earlier periods. Random splitting would be inappropriate because it can allow later conditions to inform evaluation of earlier forecasts. The central feature rule is strict: for prediction period T, historical features are constructed only from periods `< T`.

## D. Decision threshold

For State Horizon, I selected the cutoff that maximizes F1 over the pooled chronological OOF predictions (471 rows), using known training labels only. The resulting threshold is **0.3918626394013229**, with **F1 0.9829351535836177**, **precision 0.9664429530201343**, and **recall 1.0**. This threshold is a post-model decision rule; it does not alter submitted probabilities. The OOF scores and the script that reproduces these metrics are included here.

## E. Humanitarian interpretation

I interpret a model score as a risk and prioritization signal for Crisis-or-worse food insecurity. It can help responders prioritize counties for closer assessment, monitoring, preparedness, investigation, and resource-planning discussions. It does not replace IPC analysis, automatically determine humanitarian intervention, or constitute an official IPC classification.
