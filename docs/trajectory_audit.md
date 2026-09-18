# Trajectory Audit: Methodological Specification

## 1. Purpose

The trajectory audit reconstructs longitudinal participant trajectories from heterogeneous campaign data exported from GameBus and connected tools.

Its purpose is to create a transparent, reproducible representation of observed participant behavior over time that can later support case review, trajectory comparison, and personalization research.

The trajectory audit is **descriptive and rule-based**. It does not currently estimate dropout risk, adherence, motivation, intervention effectiveness, or optimal intervention adaptation.

The processing pipeline is:

```text
raw participant data
        ↓
normalized events
        ↓
participation episodes
        ↓
ParticipantState(t)
        ↓
inactivity gaps and threshold transitions
        ↓
domain/tool trajectories
        ↓
candidate trajectory patterns
        ↓
case exports and plots
```

## 2. Cohort interpretation

The preferred cohort source is the participant list contained in the campaign export.

Campaign-export membership is treated as **nominal campaign membership only**. It does not independently establish that a person was enrolled in the study, received the intervention, or had an opportunity to engage.

For this reason, the audit reports separately:

* nominal cohort size;
* participants with any observed event;
* participants with explicit engagement;
* participants without observed events;
* observed-participant coverage.

If no campaign-export cohort is available, participant IDs observed in normalized events may be used as a fallback cohort.

A participant with no observed events is therefore described as having **no observed trajectory**, rather than as a dropout or non-adherent participant.

## 3. Event-log construction

Raw exports from overlapping campaign waves are transformed into a canonical event log.

Events are assigned stable event identifiers where possible. When the same underlying event occurs in more than one wave export, it is represented once in the normalized event log while retaining provenance about all contributing waves.

Conflicting representations of the same stable event are retained as one canonical event and marked with a deduplication-conflict flag.

Rows without a stable source identifier are not assumed to be duplicates solely because their other attributes appear similar.

## 4. Behavioral engagement operationalization

The trajectory audit distinguishes several forms of evidence.

### Explicit engagement

Explicit engagement represents observed participant-initiated interaction with the intervention.

Current examples include:

* GameBus task or activity interactions;
* Nutrida intentional interactions.

Explicit engagement events reset the explicit-engagement clock used for episodes and inactivity analysis.

### Navigation evidence

Navigation events represent interaction with the interface but are not currently treated as sufficient evidence of explicit intervention engagement.

Navigation therefore does not reset the explicit-engagement clock.

### Passive behavioral sensing

Passive observations, including Garmin `DAY_AGGREGATE` events and sensor data, provide behavioral evidence but are not treated as explicit engagement.

Passive observations therefore do not bridge or terminate an explicit-engagement inactivity gap.

### Deleted activities

Activities marked as deleted are not counted as explicit engagement.

## 5. Missing-data semantics

The audit distinguishes between:

```text
available
available_empty
unavailable
```

These states have different meanings.

`available` means the stream was available and contains observations.

`available_empty` means the stream was available but contains no observations. Zero is therefore meaningful.

`unavailable` means the stream was not available for analysis. Missing observations from such a stream must not be interpreted as zero participant behavior.

Optional provider streams such as Garmin or Nutrida are not assumed to be available merely because another activity stream exists.

## 6. Observation window

The configured start of the first campaign wave is used as the default observation start.

If the first observed participant event occurs more than the configured grace period after the configured wave start, the configured start is treated as potentially stale and the effective observation start is adjusted to the first observed event.

The default grace period is:

```text
14 days
```

The analysis cutoff is normally the latest trustworthy observed event unless explicitly overridden.

An ongoing inactivity interval at the cutoff is treated as **right-censored**.

## 7. Participation episodes

Participation episodes are constructed from explicit-engagement events only.

The default technical episode-gap parameter is:

```text
14 fully inactive days
```

A new episode begins when the difference between two consecutive explicit-engagement dates is greater than 14 days.

For example:

```text
1 January → 15 January
date difference = 14 days
same episode

1 January → 16 January
date difference = 15 days
14 complete inactive days in between
new episode
```

Passive sensing or navigation during the interval does not prevent a new explicit-engagement episode from being created.

The 14-day value is an analytical parameter, not a validated clinical or behavioral threshold.

## 8. Daily ParticipantState(t)

The audit creates one daily state for every nominal cohort participant throughout the observation window.

The state representation combines:

* explicit engagement observed that day;
* navigation evidence;
* passive behavioral sensing;
* stream availability;
* previous explicit-engagement history;
* time since last explicit engagement.

Current engagement-state labels include:

* `no_explicit_engagement_observed_yet`;
* `active`;
* `quiet`;
* `prolonged_inactivity_7d`;
* `prolonged_inactivity_14d`;
* `unavailable`.

`no_explicit_engagement_observed_yet` is intentionally observational. It does not imply that the participant was verified as enrolled but failed to start.

## 9. Inactivity gaps and threshold transitions

An inactivity gap represents time without observed explicit engagement after explicit engagement has occurred.

Closed gaps end when a later explicit-engagement event is observed.

Trailing gaps that remain open at the analysis cutoff are marked as right-censored.

The audit currently reports exploratory inactivity thresholds of:

```text
7 days
14 days
21 days
```

For each threshold, the audit records:

* number of gaps reaching the threshold;
* number followed by observed re-engagement;
* number still ongoing at the cutoff.

These thresholds are **sensitivity scales**, not validated dropout, adherence, or risk thresholds.

Observed re-engagement proportions are therefore descriptive and may be lower bounds because some ongoing gaps are censored before their eventual outcome is known.

## 10. Domain and tool trajectories

Activity events are mapped to intervention domains using campaign configuration where possible.

Current controlled domains are:

* nutrition;
* physical activity;
* mental wellbeing.

Campaign configuration mappings are preferred over heuristic fallback mappings.

Events may belong to multiple domains. In such cases, event weight is divided equally across assigned domains so that one event contributes a total weight of one.

Tool/provider trajectories currently distinguish sources such as:

* GameBus Studio;
* Garmin;
* Nutrida.

Points from different tools should not automatically be interpreted as directly comparable behavioral quantities.

## 11. Candidate trajectory patterns

The audit generates transparent, rule-based **candidate trajectory patterns** for later human review.

Current examples include:

* `no_explicit_engagement_observed_by_cutoff`;
* `navigation_without_explicit_engagement`;
* `behavioral_sensor_without_explicit_engagement`;
* `current_prolonged_inactivity_7d`;
* `current_prolonged_inactivity_14d`;
* `reengaged_after_long_gap`;
* `repeated_long_gaps`;
* `recent_engagement_decline`;
* `selective_domain_disappearance`;
* `selective_tool_disappearance`.

These are candidate queries over observed trajectories.

They are **not classifiers** and should not be interpreted as:

* dropout prediction;
* adherence classification;
* psychological disengagement;
* motivation;
* intervention failure;
* causal effectiveness;
* clinical risk.

Candidate behavioral patterns are suppressed where core trajectory evidence is insufficient.

## 12. Participant data-quality state

Participant core trajectory evidence is summarized using four states.

| State                              | Interpretation                                                                                                 |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `sufficient_for_core_trajectory`   | Required core data are available and participant events are observed without identified core quality problems. |
| `core_trajectory_with_cautions`    | Participant events are observed, but quality flags require caution.                                            |
| `no_observed_trajectory`           | Core source data and cohort membership are available, but no participant events are observed.                  |
| `insufficient_for_core_trajectory` | Required core source data or cohort information are unavailable.                                               |

`no_observed_trajectory` is deliberately distinct from `insufficient_for_core_trajectory`.

The first is an observation within available data; the second is a limitation of the available evidence.

## 13. Construct specification

| Construct                   | Operational definition                                                        | Main source                   | Parameter                                     | Interpretation boundary                                      | Current validation                 |
| --------------------------- | ----------------------------------------------------------------------------- | ----------------------------- | --------------------------------------------- | ------------------------------------------------------------ | ---------------------------------- |
| Nominal cohort membership   | Participant ID present in campaign export                                     | Campaign aggregation export   | None                                          | Does not prove study enrollment                              | Cross-campaign consistency checked |
| Observed participation      | At least one normalized event                                                 | Normalized event log          | None                                          | Does not necessarily imply explicit intervention interaction | Cross-campaign consistency checked |
| Explicit engagement         | Eligible intentional activity event that is not deleted or passive            | Activity events               | Rule based                                    | Does not measure psychological engagement                    | Unit tested                        |
| Passive behavioral evidence | Garmin `DAY_AGGREGATE` or sensor observation                                  | Activity/sensor streams       | Rule based                                    | Does not reset explicit-engagement clock                     | Unit tested                        |
| Participation episode       | Sequence of explicit-engagement events without crossing episode-gap boundary  | Explicit-engagement log       | 14-day default                                | Technical segmentation, not behavioral phase classification  | Boundary unit tested               |
| Inactivity gap              | Calendar days following explicit engagement without later explicit engagement | Explicit-engagement log       | 7/14/21-day sensitivity scales                | Not equivalent to dropout                                    | Unit tested                        |
| Right censoring             | Inactivity continues when observation window ends                             | Observation window + gaps     | Cutoff date                                   | Final outcome unknown                                        | Unit tested                        |
| Daily participant state     | Daily aggregation of explicit, passive, navigation and availability evidence  | Multiple streams              | Daily resolution                              | Descriptive state representation                             | Unit tested                        |
| Domain trajectory           | Activity evidence mapped to configured intervention domains                   | Campaign description + events | Equal-share weighting for multi-domain events | Points not necessarily comparable across providers           | Cross-campaign mapping checked     |
| Candidate pattern           | Transparent rule-based query over trajectory evidence                         | State, gaps, domains/tools    | Pattern-specific                              | Not predictive classification                                | Semantic unit tested               |

## 14. Cross-campaign validation

The trajectory audit has been executed on campaigns:

```text
283
294
379
456
462
```

All five campaigns complete the full audit pipeline and produce:

* normalized event logs;
* participant episodes;
* daily participant states;
* inactivity gaps;
* inactivity-threshold transitions;
* domain/tool trajectories;
* participant data-quality states;
* candidate patterns;
* case exports;
* audit summaries;
* audit manifests.

Cross-campaign validation also confirms that domain mapping currently produces zero unmapped activity events in all five evaluated campaigns.

Observed participant coverage differs substantially between campaigns. This is an important reason to report nominal cohort membership separately from observed participation.

Campaign 379 is the clearest example: only a small subset of nominal campaign members has any observed trajectory. The current data do not establish whether participants without events were actually enrolled or exposed to the intervention, so their absence of events must not be interpreted as dropout.

## 15. Validation strategy

Current validation combines:

1. **Semantic unit tests** for engagement classification, missing-stream behavior, episode boundaries, inactivity gaps, right censoring, event deduplication, participant data-quality classification, and candidate-pattern semantics.

2. **Cross-campaign execution validation** across heterogeneous campaign exports.

3. **Case-level exports and visualizations** supporting manual inspection of selected trajectories.

4. **Audit manifests** recording configuration, input hashes, and generated artifacts.

Future validation should include expert review of selected trajectory cases and assessment of whether the reconstructed states are useful for intervention-design and personalization decisions.

## 16. Limitations

The trajectory audit reconstructs only what is observable in the available digital traces.

Important limitations include:

* campaign-export membership may not equal verified study enrollment;
* absence of observed events does not prove absence of behavior outside the observed system;
* unavailable streams cannot be interpreted as behavioral zeros;
* system usage is not equivalent to psychological engagement;
* current inactivity thresholds are exploratory;
* provider-specific points may have different meanings;
* candidate patterns have not been validated as predictive or clinical signals;
* right-censored inactivity intervals have unknown future outcomes;
* current trajectory states do not establish causal relationships between intervention exposure and participant behavior.

The resulting trajectory representation should therefore be interpreted as a transparent evidence layer for subsequent human review and future personalization methods, rather than as a behavioral or clinical ground truth.
