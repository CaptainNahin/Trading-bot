#!/bin/bash
echo "Starting QuantEdge AI Deployment Pipeline..."

# 1. Supabase setup instructions
echo "----------------------------------------------------"
echo "SUPABASE SETUP REQUIRED:"
echo "Before deploying to Vercel, please ensure your Supabase database is ready."
echo "If you have a Supabase connection string, run:"
echo "alembic upgrade head"
echo "----------------------------------------------------"

# 2. Vercel deployment
echo "Deploying to Vercel..."
npx vercel deploy --prod --yes

echo "Deployment finished! Don't forget to add your environment variables in the Vercel Dashboard:"
echo "- AGENTROUTER_API_KEY"
echo "- SUPABASE_URL"
echo "- SUPABASE_SERVICE_ROLE_KEY"
echo "- TWELVE_DATA_API_KEY"
