#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ChromaDB Configuration Module

This module provides configuration functions to suppress ChromaDB telemetry
warnings and optimize the database experience for the Infomaid RAG system.
It handles environment variable setup, logging configuration, and warning
suppression to create a cleaner user experience.

Key Features:
    - Telemetry suppression for privacy
    - Warning and logging configuration
    - Environment variable management
    - Automatic configuration on import

Purpose:
    ChromaDB by default enables telemetry and can generate various warnings
    that clutter the console output. This module ensures a clean, private
    operation of the vector database by disabling unnecessary feedback.

Usage:
    from infomaid.chromadb_config import configure_chromadb
    configure_chromadb()  # Manual configuration
    
    # Or simply import the module for automatic configuration:
    import infomaid.chromadb_config
"""

import os
import logging
import warnings

def configure_chromadb():
    """
    Configure ChromaDB to suppress telemetry and warnings for clean operation.
    
    This function sets up the optimal ChromaDB environment by:
    1. Disabling telemetry collection for privacy
    2. Suppressing unnecessary warning messages
    3. Configuring logging levels to reduce console clutter
    4. Setting authentication provider variables
    
    Environment Variables Set:
        - ANONYMIZED_TELEMETRY: Disabled to prevent data collection
        - CHROMA_CLIENT_AUTH_PROVIDER: Cleared for local operation
        - CHROMA_SERVER_AUTH_PROVIDER: Cleared for local operation
        
    Logging Configuration:
        - ChromaDB general logging set to ERROR level
        - Telemetry logging set to CRITICAL level
        - UserWarnings from ChromaDB module filtered
        
    Side Effects:
        - Modifies global environment variables
        - Changes logging configuration
        - Filters warning messages
        - Attempts to configure ChromaDB settings if available
        
    Returns:
        None
    """
    
    # Suppress telemetry collection for privacy and performance
    os.environ['ANONYMIZED_TELEMETRY'] = 'False'
    os.environ['CHROMA_CLIENT_AUTH_PROVIDER'] = ''
    os.environ['CHROMA_SERVER_AUTH_PROVIDER'] = ''
    
    # Configure logging levels to reduce console output
    logging.getLogger('chromadb').setLevel(logging.ERROR)
    logging.getLogger('chromadb.telemetry').setLevel(logging.CRITICAL)
    
    # Suppress warning messages for cleaner console output
    warnings.filterwarnings('ignore', category=UserWarning, module='chromadb')
    warnings.filterwarnings('ignore', message='.*telemetry.*')
    
    # Attempt to disable telemetry at the ChromaDB module level
    try:
        import chromadb
        if hasattr(chromadb, 'config'):
            chromadb.config.Settings(anonymized_telemetry=False)
    except:
        # Gracefully handle any import or configuration errors
        pass

# Automatically configure ChromaDB when this module is imported
configure_chromadb()
