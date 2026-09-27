"""Initial DemandIQ application tables.

Revision ID: 0001_initial
Revises: None
"""
from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table("users", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("email", sa.String(255), nullable=False), sa.Column("name", sa.String(120)), sa.Column("password_hash", sa.String(512), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False), sa.UniqueConstraint("email"))
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_table("datasets", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("owner_id", sa.Integer(), sa.ForeignKey("users.id")), sa.Column("name", sa.String(255), nullable=False), sa.Column("row_count", sa.Integer(), nullable=False), sa.Column("source", sa.String(80), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))
    op.create_index("ix_datasets_owner_id", "datasets", ["owner_id"])
    op.create_table("prediction_runs", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("owner_id", sa.Integer(), sa.ForeignKey("users.id")), sa.Column("dataset_id", sa.Integer(), sa.ForeignKey("datasets.id")), sa.Column("mode", sa.String(40), nullable=False), sa.Column("status", sa.String(40), nullable=False), sa.Column("row_count", sa.Integer(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))
    op.create_index("ix_prediction_runs_owner_id", "prediction_runs", ["owner_id"])
    op.create_table("predictions", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("run_id", sa.Integer(), sa.ForeignKey("prediction_runs.id")), sa.Column("store_nbr", sa.Integer(), nullable=False), sa.Column("family", sa.String(120), nullable=False), sa.Column("predicted_demand", sa.Float(), nullable=False), sa.Column("high_demand_probability", sa.Float(), nullable=False), sa.Column("is_high_demand", sa.Boolean(), nullable=False), sa.Column("input_payload", sa.JSON(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))
    op.create_index("ix_predictions_run_id", "predictions", ["run_id"]); op.create_index("ix_predictions_store_nbr", "predictions", ["store_nbr"]); op.create_index("ix_predictions_family", "predictions", ["family"])
    op.create_table("anomalies", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("store_nbr", sa.Integer(), nullable=False), sa.Column("family", sa.String(120), nullable=False), sa.Column("anomaly_score", sa.Float(), nullable=False), sa.Column("anomaly_type", sa.String(80), nullable=False), sa.Column("observed_payload", sa.JSON()), sa.Column("created_at", sa.DateTime(timezone=True)))
    op.create_index("ix_anomalies_store_nbr", "anomalies", ["store_nbr"]); op.create_index("ix_anomalies_family", "anomalies", ["family"])
    op.create_table("segment_results", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("entity_type", sa.String(40), nullable=False), sa.Column("entity_key", sa.String(120), nullable=False), sa.Column("cluster", sa.Integer(), nullable=False), sa.Column("metrics", sa.JSON(), nullable=False))
    op.create_index("ix_segment_results_entity_type", "segment_results", ["entity_type"]); op.create_index("ix_segment_results_entity_key", "segment_results", ["entity_key"])
    op.create_table("replenishment_recommendations", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("store_nbr", sa.Integer(), nullable=False), sa.Column("family", sa.String(120), nullable=False), sa.Column("priority_score", sa.Float(), nullable=False), sa.Column("priority", sa.String(20), nullable=False), sa.Column("action", sa.String(120), nullable=False), sa.Column("payload", sa.JSON(), nullable=False))
    op.create_index("ix_replenishment_recommendations_store_nbr", "replenishment_recommendations", ["store_nbr"]); op.create_index("ix_replenishment_recommendations_family", "replenishment_recommendations", ["family"]); op.create_index("ix_replenishment_recommendations_priority", "replenishment_recommendations", ["priority"]); op.create_index("ix_replenishment_recommendations_priority_score", "replenishment_recommendations", ["priority_score"])
    op.create_table("model_versions", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("name", sa.String(120), nullable=False), sa.Column("version", sa.String(80), nullable=False), sa.Column("feature_schema", sa.JSON(), nullable=False), sa.Column("library_versions", sa.JSON(), nullable=False), sa.Column("notes", sa.Text()), sa.UniqueConstraint("name"))

def downgrade():
    for table in ["model_versions", "replenishment_recommendations", "segment_results", "anomalies", "predictions", "prediction_runs", "datasets", "users"]:
        op.drop_table(table)
