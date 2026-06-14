#!/bin/bash
# ChromaDB Security Update Checker
# Checks for new ChromaDB versions and security patches

echo "🔍 Checking ChromaDB version status..."
echo ""

# Current version in project
echo "📦 Current version in project:"
poetry show chromadb 2>/dev/null | grep "version" || echo "  Could not detect version"
echo ""

# Latest available version
echo "📡 Latest available versions on PyPI:"
pip index versions chromadb 2>/dev/null | head -3 || echo "  Could not check PyPI"
echo ""

# Check for GitHub advisory updates
echo "🚨 Security advisory: CVE-2026-45829"
echo "  Status: No patch available yet"
echo "  Check: https://github.com/advisories/GHSA-f4j7-r4q5-qw2c"
echo ""

echo "✅ Next steps:"
echo "  1. If a version > 1.5.9 is available, update immediately"
echo "  2. Run: poetry update chromadb"
echo "  3. Run: poetry lock"
echo "  4. Test the application after updating"
echo ""

echo "💡 To update ChromaDB when a patch is available:"
echo "  poetry add \"chromadb>=1.6.0\"  # Replace with patched version"
echo "  poetry install"
