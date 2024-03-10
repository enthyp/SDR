# SDR

### Radio-Data-System
`rds_fundamentals` directory is for exploration of various topics related to RDS (and digital communication in general), like PLLs, timing recovery, channel coding. It's messy!

`rds_experiments` directory uses building blocks worked out in `rds_fundamentals` to decode real-life signals and see how e.g. carrier recovery technique affects the results. All the stuff in `rds.py` has been worked out in `rds_fundamentals` first, except for block synchronization and RadioText decoding since they are very RDS-specific.

To run any of the notebooks you need to run Jupyter in the directory, otherwise imports won't work.
