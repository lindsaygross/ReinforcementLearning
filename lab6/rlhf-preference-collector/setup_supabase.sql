CREATE TABLE IF NOT EXISTS public.preference_data (
    id UUID PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    prompt TEXT NOT NULL,
    response_a TEXT NOT NULL,
    response_b TEXT NOT NULL,
    chosen TEXT NULL,
    rejected TEXT NULL,
    preference TEXT NOT NULL CHECK (preference IN ('a', 'b', 'tie')),
    model TEXT NOT NULL,
    generation_params JSONB NOT NULL,
    response_a_latency_ms INTEGER NOT NULL,
    response_b_latency_ms INTEGER NOT NULL,
    session_id UUID NOT NULL,
    position_mapping JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_preference_data_timestamp
    ON public.preference_data (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_preference_data_session_id
    ON public.preference_data (session_id);
