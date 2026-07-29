from datetime import timedelta
from feast import FeatureView, Field, PushSource, FileSource
from feast.types import Int64, Float32, Bool
from entities import user

user_behavior_batch_source = FileSource(
    name="user_behavior_batch_source",
    path="../data/user_behavior_features.parquet",
    timestamp_field="event_timestamp",
)

user_behavior_push_source = PushSource(
    name="user_behavior_push_source",
    batch_source=user_behavior_batch_source,
)

user_behavior_fv = FeatureView(
    name="user_behavior_features",
    entities=[user],
    ttl=timedelta(minutes=30),
    schema=[
        Field(name="txn_count_5min", dtype=Int64),
        Field(name="avg_amount_5min", dtype=Float32),
        Field(name="amount_ratio", dtype=Float32),
        Field(name="velocity_flag", dtype=Bool),
        Field(name="amount_spike_flag", dtype=Bool),
    ],
    source=user_behavior_push_source,
    online=True,
)
