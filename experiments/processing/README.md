# Processing and simulation

`materialize.py` reconstructs pinned dependencies and applies release patches. `safelibero/live_viewer.py` is the authoritative experiment/evaluator source. It is installed into the generated OpenPI checkout by the materializer so upstream imports and the separate simulator environment remain supported.

Run commands are documented in the repository README. Keep run outputs outside tracked source and record interventions and effective parameters using the experiment template.
