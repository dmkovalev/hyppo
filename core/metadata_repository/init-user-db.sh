#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE USER core;
    CREATE DATABASE metadata_repository;
    GRANT ALL PRIVILEGES ON DATABASE metadata_repository TO core;
EOSQL
