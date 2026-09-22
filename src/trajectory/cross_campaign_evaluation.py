from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.trajectory.dynamic_event_history import (
	DEFAULT_DYNAMIC_FEATURES,
	DiscreteTimeHazardModel,
	build_reengagement_hazard_dataset,
	fit_discrete_time_hazard_model,
	predict_reengagement_hazard,
)
from src.trajectory.dynamic_review_trigger import (
	apply_dynamic_review_trigger,
	calibrate_dynamic_review_trigger,
	save_dynamic_review_trigger,
)


@dataclass(
	frozen=True
)
class CampaignAudit:
	audit_dir: Path
	campaign_id: str
	campaign_abbreviation: str | None
	source_type: str | None


EVALUATION_COLUMNS = [
	"evaluation_scope",
	"campaign_id",
	"campaign_abbreviation",

	"rows",
	"participants",

	"reengagement_events",
	"reengagement_event_rate",

	"brier_score",
	"log_loss",
	"roc_auc",
]


def _safe_float(
	value,
) -> float | None:

	if value is None:
		return None

	number = float(
		value
	)

	if not np.isfinite(
		number
	):
		return None

	return number


def _read_audit(
	audit_dir: str | Path,
) -> tuple[
	CampaignAudit,
	pd.DataFrame,
]:
	"""
	Read one immutable trajectory-audit run.

	Required files:

		run_manifest.json
		results/participant_day_modeling.csv
	"""

	audit_path = Path(
		audit_dir
	).expanduser().resolve()

	manifest_path = (
		audit_path
		/ "run_manifest.json"
	)

	modeling_path = (
		audit_path
		/ "results"
		/ "participant_day_modeling.csv"
	)

	if not manifest_path.is_file():
		raise ValueError(
			"Audit run is missing "
			f"{manifest_path.name}: "
			f"{audit_path}"
		)

	if not modeling_path.is_file():
		raise ValueError(
			"Audit run is missing "
			"participant_day_modeling.csv: "
			f"{audit_path}"
		)

	manifest = json.loads(
		manifest_path.read_text(
			encoding="utf-8"
		)
	)

	campaign = manifest.get(
		"campaign",
		{},
	)

	campaign_id = str(
		campaign.get(
			"id",
			""
		)
	).strip()

	if not campaign_id:
		raise ValueError(
			"Audit run does not contain a "
			"campaign ID: "
			f"{audit_path}"
		)

	abbreviation = campaign.get(
		"abbreviation"
	)

	if abbreviation is not None:
		abbreviation = str(
			abbreviation
		).strip() or None

	audit = CampaignAudit(
		audit_dir=audit_path,
		campaign_id=campaign_id,
		campaign_abbreviation=(
			abbreviation
		),
		source_type=manifest.get(
			"source_type"
		),
	)

	modeling = pd.read_csv(
		modeling_path
	)

	return (
		audit,
		modeling,
	)


def _roc_auc(
	observed: pd.Series,
	predicted: pd.Series,
) -> float | None:
	"""
	ROC AUC using the rank-sum formulation.

	Returns None when only one outcome class is
	present.
	"""

	observed = pd.to_numeric(
		observed,
		errors="coerce",
	)

	predicted = pd.to_numeric(
		predicted,
		errors="coerce",
	)

	valid = (
		observed.isin(
			[
				0,
				1,
			]
		)
		& predicted.notna()
	)

	observed = observed.loc[
		valid
	].astype(
		int
	)

	predicted = predicted.loc[
		valid
	].astype(
		float
	)

	positives = int(
		observed.sum()
	)

	negatives = int(
		len(
			observed
		)
		- positives
	)

	if (
		positives == 0
		or negatives == 0
	):
		return None

	ranks = predicted.rank(
		method="average"
	)

	positive_rank_sum = float(
		ranks.loc[
			observed.eq(
				1
			)
		].sum()
	)

	auc = (
		positive_rank_sum
		- (
			positives
			* (
				positives
				+ 1
			)
			/ 2
		)
	) / (
		positives
		* negatives
	)

	return float(
		auc
	)


def _evaluate_probabilities(
	frame: pd.DataFrame,
	*,
	evaluation_scope: str,
	campaign_id: str,
	campaign_abbreviation: str | None,
) -> dict[str, object]:

	if frame.empty:

		return {
			"evaluation_scope": (
				evaluation_scope
			),
			"campaign_id": campaign_id,
			"campaign_abbreviation": (
				campaign_abbreviation
			),
			"rows": 0,
			"participants": 0,
			"reengagement_events": 0,
			"reengagement_event_rate": (
				None
			),
			"brier_score": None,
			"log_loss": None,
			"roc_auc": None,
		}

	observed = pd.to_numeric(
		frame[
			"explicit_engagement_next_day"
		],
		errors="coerce",
	)

	predicted = pd.to_numeric(
		frame[
			(
				"predicted_next_day_"
				"reengagement_probability"
			)
		],
		errors="coerce",
	)

	valid = (
		observed.isin(
			[
				0,
				1,
			]
		)
		& predicted.notna()
	)

	observed = observed.loc[
		valid
	].astype(
		float
	)

	predicted = (
		predicted.loc[
			valid
		]
		.astype(
			float
		)
		.clip(
			1e-9,
			1.0 - 1e-9,
		)
	)

	n = len(
		observed
	)

	if n == 0:

		return {
			"evaluation_scope": (
				evaluation_scope
			),
			"campaign_id": campaign_id,
			"campaign_abbreviation": (
				campaign_abbreviation
			),
			"rows": 0,
			"participants": 0,
			"reengagement_events": 0,
			"reengagement_event_rate": (
				None
			),
			"brier_score": None,
			"log_loss": None,
			"roc_auc": None,
		}

	events = int(
		observed.sum()
	)

	brier = float(
		np.mean(
			(
				predicted
				- observed
			)
			** 2
		)
	)

	log_loss = float(
		-np.mean(
			(
				observed
				* np.log(
					predicted
				)
			)
			+ (
				(
					1.0
					- observed
				)
				* np.log(
					1.0
					- predicted
				)
			)
		)
	)

	valid_indices = observed.index

	return {
		"evaluation_scope": (
			evaluation_scope
		),

		"campaign_id": campaign_id,

		"campaign_abbreviation": (
			campaign_abbreviation
		),

		"rows": n,

		"participants": int(
			frame.loc[
				valid_indices,
				"participant_id",
			].nunique()
		),

		"reengagement_events": (
			events
		),

		"reengagement_event_rate": (
			events
			/ n
		),

		"brier_score": brier,

		"log_loss": log_loss,

		"roc_auc": _roc_auc(
			observed,
			predicted,
		),
	}


def _model_payload(
	model: DiscreteTimeHazardModel,
) -> dict[str, object]:

	return {
		"model_type": (
			"pooled_logistic_discrete_time_"
			"reengagement_hazard"
		),

		"feature_names": list(
			model.feature_names
		),

		"intercept": (
			model.intercept
		),

		"coefficients": list(
			model.coefficients
		),

		"imputation_values": list(
			model.imputation_values
		),

		"means": list(
			model.means
		),

		"scales": list(
			model.scales
		),

		"l2_penalty": (
			model.l2_penalty
		),
	}


def run_cross_campaign_dynamic_evaluation(
	*,
	development_audit_dirs: Iterable[
		str | Path
	],
	evaluation_audit_dirs: Iterable[
		str | Path
	],
	output_dir: str | Path,
	feature_names: tuple[
		str,
		...
	] = DEFAULT_DYNAMIC_FEATURES,
	l2_penalty: float = 0.10,
) -> dict[str, object]:
	"""
	Fit the dynamic event-history model using only
	development campaigns and evaluate the frozen
	model on separate campaign audits.

	Evaluation campaigns never contribute to:

		coefficient estimation
		missing-value imputation
		means
		scales

	This is the experiment-level workflow intended
	for the paper.
	"""

	development_paths = [
		Path(
			path
		).expanduser().resolve()
		for path in development_audit_dirs
	]

	evaluation_paths = [
		Path(
			path
		).expanduser().resolve()
		for path in evaluation_audit_dirs
	]

	if not development_paths:
		raise ValueError(
			"At least one development audit "
			"is required."
		)

	if not evaluation_paths:
		raise ValueError(
			"At least one evaluation audit "
			"is required."
		)

	overlap = (
		set(
			development_paths
		)
		& set(
			evaluation_paths
		)
	)

	if overlap:
		raise ValueError(
			"The same audit run cannot be used "
			"for both development and evaluation."
		)

	output_path = Path(
		output_dir
	).expanduser().resolve()

	output_path.mkdir(
		parents=True,
		exist_ok=True,
	)

	# -------------------------------------------------
	# Development campaigns
	# -------------------------------------------------

	development_metadata = []
	development_frames = []

	for path in development_paths:

		audit, modeling = _read_audit(
			path
		)

		risk_set = (
			build_reengagement_hazard_dataset(
				modeling
			)
		)

		if risk_set.empty:
			raise ValueError(
				"Development campaign has no "
				"eligible re-engagement risk-set "
				f"rows: {audit.campaign_id}"
			)

		risk_set.insert(
			0,
			"campaign_id",
			audit.campaign_id,
		)

		risk_set.insert(
			1,
			"campaign_abbreviation",
			audit.campaign_abbreviation,
		)

		development_frames.append(
			risk_set
		)

		development_metadata.append(
			audit
		)

	training = pd.concat(
		development_frames,
		ignore_index=True,
	)

	model = (
		fit_discrete_time_hazard_model(
			training,
			feature_names=(
				feature_names
			),
			l2_penalty=(
				l2_penalty
			),
		)
	)

	development_predictions = (
		predict_reengagement_hazard(
			model,
			training,
		)
	)

	(
		trigger_config,
		trigger_calibration,
	) = calibrate_dynamic_review_trigger(
		development_predictions,
		outcome_horizon_days=14,
		persistence_days=3,
	)

	development_predictions = (
		apply_dynamic_review_trigger(
			development_predictions,
			trigger_config,
		)
	)

	development_prediction_path = (
		output_path
		/ (
			"dynamic_event_history_"
			"development_predictions.csv"
		)
	)

	development_predictions.to_csv(
		development_prediction_path,
		index=False,
	)

	save_dynamic_review_trigger(
		output_dir=str(
			output_path
		),
		config=trigger_config,
		calibration=trigger_calibration,
	)

	# -------------------------------------------------
	# Frozen model artifact
	# -------------------------------------------------

	model_path = (
		output_path
		/ "dynamic_event_history_model.json"
	)

	model_path.write_text(
		json.dumps(
			_model_payload(
				model
			),
			indent=2,
		),
		encoding="utf-8",
	)

	coefficient_table = (
		model.coefficient_table()
	)

	coefficient_path = (
		output_path
		/ "dynamic_event_history_coefficients.csv"
	)

	coefficient_table.to_csv(
		coefficient_path,
		index=False,
	)

	# -------------------------------------------------
	# Evaluation campaigns
	# -------------------------------------------------

	evaluation_metadata = []
	prediction_frames = []
	metric_rows = []

	for path in evaluation_paths:

		audit, modeling = _read_audit(
			path
		)

		risk_set = (
			build_reengagement_hazard_dataset(
				modeling
			)
		)

		prediction = (
			predict_reengagement_hazard(
				model,
				risk_set,
			)
		)

		prediction.insert(
			0,
			"campaign_id",
			audit.campaign_id,
		)

		prediction.insert(
			1,
			"campaign_abbreviation",
			audit.campaign_abbreviation,
		)

		prediction = (
			apply_dynamic_review_trigger(
				prediction,
				trigger_config,
			)
		)

		prediction_frames.append(
			prediction
		)

		metric_rows.append(
			_evaluate_probabilities(
				prediction,
				evaluation_scope=(
					"campaign"
				),
				campaign_id=(
					audit.campaign_id
				),
				campaign_abbreviation=(
					audit.campaign_abbreviation
				),
			)
		)

		evaluation_metadata.append(
			audit
		)

	predictions = pd.concat(
		prediction_frames,
		ignore_index=True,
	)

	prediction_path = (
		output_path
		/ "dynamic_event_history_predictions.csv"
	)

	predictions.to_csv(
		prediction_path,
		index=False,
	)

	# -------------------------------------------------
	# Pooled evaluation
	# -------------------------------------------------

	metric_rows.append(
		_evaluate_probabilities(
			predictions,
			evaluation_scope=(
				"pooled_evaluation"
			),
			campaign_id="ALL",
			campaign_abbreviation=None,
		)
	)

	evaluation = pd.DataFrame(
		metric_rows
	)

	for column in EVALUATION_COLUMNS:
		if column not in evaluation.columns:
			evaluation[column] = pd.NA

	evaluation = evaluation[
		EVALUATION_COLUMNS
	]

	evaluation_path = (
		output_path
		/ "dynamic_event_history_evaluation.csv"
	)

	evaluation.to_csv(
		evaluation_path,
		index=False,
	)

	# -------------------------------------------------
	# Reproducibility manifest
	# -------------------------------------------------

	experiment_manifest = {
		"schema_version": 1,

		"method": (
			"pooled_logistic_discrete_time_"
			"reengagement_hazard"
		),

		"target": (
			"explicit_engagement_next_day"
		),

		"risk_set": (
			"maintenance/reengagement days "
			"with days_since_last_explicit_"
			"engagement >= 1 and observable "
			"next-day outcome"
		),

		"feature_names": list(
			feature_names
		),

		"l2_penalty": (
			l2_penalty
		),

		"development_campaigns": [
			{
				**asdict(
					audit
				),
				"audit_dir": str(
					audit.audit_dir
				),
			}
			for audit in development_metadata
		],

		"evaluation_campaigns": [
			{
				**asdict(
					audit
				),
				"audit_dir": str(
					audit.audit_dir
				),
			}
			for audit in evaluation_metadata
		],

		"outputs": {
			"model": (
				model_path.name
			),
			"coefficients": (
				coefficient_path.name
			),
			"predictions": (
				prediction_path.name
			),
			"evaluation": (
				evaluation_path.name
			),
		},
		"development_predictions": (
			development_prediction_path.name
		),

		"review_trigger": (
			"dynamic_review_trigger.json"
		),

		"review_trigger_calibration": (
			"dynamic_review_trigger_calibration.csv"
		),

		"review_trigger": {
			"calibration_population": (
				"development campaigns only"
			),
			"calibration_outcome": (
				"continued_inactivity_14d"
			),
			"persistence_days": (
				trigger_config.persistence_days
			),
			"inactivity_risk_threshold": (
				trigger_config
				.inactivity_risk_threshold
			),
		},
	}

	manifest_path = (
		output_path
		/ "cross_campaign_experiment.json"
	)

	manifest_path.write_text(
		json.dumps(
			experiment_manifest,
			indent=2,
		),
		encoding="utf-8",
	)

	return {
		"model": model,
		"training": training,
		"predictions": predictions,
		"evaluation": evaluation,
		"coefficients": (
			coefficient_table
		),
		"manifest_path": (
			manifest_path
		),
		"development_predictions": (
			development_predictions
		),
		"trigger_config": (
			trigger_config
		),
		"trigger_calibration": (
			trigger_calibration
		),
	}