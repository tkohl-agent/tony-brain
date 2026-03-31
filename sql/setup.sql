create extension if not exists vector;

create table memories (
  id uuid primary key default gen_random_uuid(),
  type text,
  content text,
  embedding vector(1536),
  importance float default 0.5,
  created_at timestamp default now(),
  last_used timestamp default now()
);

alter table memories
add constraint valid_type
check (type in ('fact', 'preference', 'goal', 'project', 'relationship'));

create or replace function match_memories (
  query_embedding vector(1536),
  match_threshold float,
  match_count int
)
returns table (
  id uuid,
  content text,
  type text,
  similarity float
)
language sql
as $$
  select
    id,
    content,
    type,
    (1 - (embedding <=> query_embedding)) * importance as similarity
  from memories
  where (1 - (embedding <=> query_embedding)) > match_threshold
  order by similarity desc
  limit match_count;
$$;

create or replace function decay_memory()
returns void
language sql
as $$
  update memories
  set importance = importance * 0.98
  where created_at < now() - interval '1 day';
$$;
