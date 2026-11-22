"""Connect to the lab's DataJoint database and load the tracking schema.

DataJoint 2.x reads its settings, credentials included, from datajoint.json in
the project root (copied from MazePipeline/DJ_EPHYS_BPOD-main's
dj_local_conf.json, and gitignored) when it is imported, so nothing needs
loading here.

    python test_dj_connection.py
"""

import datajoint as dj

tracking = dj.VirtualModule('tracking', 'noamjoshua_TRACKING')

# session dates live in the experiment schema, not in tracking
exp = dj.VirtualModule('exp', 'noamjoshua_exp2')

# every tracked trial; each row's key starts with the mouse's subject_id
trials = tracking.TrackingTrial

# a query over the subject_id column alone, one row per distinct value
all_mouse_ids = dj.U('subject_id')

# keep only the ids that appear in the tracked trials
tracked_mouse_ids = all_mouse_ids & trials

# nothing has been downloaded until now: fetch runs the query and returns an array
mice = tracked_mouse_ids.fetch('subject_id')

for mouse in mice:
    # this mouse's tracked trials
    mouse_trials = trials & {'subject_id': mouse}

    # the sessions those trials belong to: & matches on the shared key
    # columns, subject_id and session
    mouse_sessions = exp.Session & mouse_trials

    # download just the dates, earliest first
    dates = mouse_sessions.fetch('session_date', order_by='session_date')

    print(f'mouse {mouse}: {len(dates)} tracked sessions')
    for date in dates:
        print('   ', date)

# the three sessions the analysis uses (pipeline.py's sessions), all from mouse 108104
ANALYSIS_MOUSE = 108104
ANALYSIS_DATES = ['2026-05-17', '2026-05-18', '2026-05-25']

# that mouse's tracked sessions, then just the ones on the analysis dates
tracked_sessions = exp.Session & (trials & {'subject_id': ANALYSIS_MOUSE})
analysis_sessions = tracked_sessions & [{'session_date': date} for date in ANALYSIS_DATES]

# download the date and the session number DataJoint knows each one by
dates, numbers = analysis_sessions.fetch('session_date', 'session', order_by='session_date')

print()
print(f'analysis sessions (mouse {ANALYSIS_MOUSE}):')
for date, number in zip(dates, numbers):
    print(f'    {date}  (session {number})')
