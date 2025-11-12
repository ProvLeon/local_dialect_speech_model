'use client';

import React, { useState, useEffect } from 'react';
import { Bug, ChevronDown, ChevronUp, Globe, Server, CheckCircle, XCircle, AlertTriangle } from 'lucide-react';
import { getConfigSummary, shouldSkipLocalAPI } from '../lib/api-wrapper';
import { isGradioEnabled, getGradioConfig } from '../lib/gradio-client';

interface DebugInfo {
  timestamp: string;
  config: any;
  environment: any;
  apiStatus: {
    mode: string;
    gradioEnabled: boolean;
    skipLocal: boolean;
  };
}

export default function DebugPanel() {
  const [isOpen, setIsOpen] = useState(false);
  const [debugInfo, setDebugInfo] = useState<DebugInfo | null>(null);
  const [refreshCount, setRefreshCount] = useState(0);

  // Only show debug panel in debug mode
  const showPanel = process.env.NEXT_PUBLIC_DEBUG_MODE === 'true';

  useEffect(() => {
    if (showPanel) {
      refreshDebugInfo();
    }
  }, [showPanel, refreshCount]);

  const refreshDebugInfo = () => {
    const config = getConfigSummary();
    const gradioConfig = getGradioConfig();

    const info: DebugInfo = {
      timestamp: new Date().toLocaleTimeString(),
      config,
      environment: {
        USE_GRADIO: process.env.NEXT_PUBLIC_USE_GRADIO,
        GRADIO_SPACE_ID: process.env.NEXT_PUBLIC_GRADIO_SPACE_ID,
        API_BASE_URL: process.env.NEXT_PUBLIC_API_BASE_URL,
        DEBUG_MODE: process.env.NEXT_PUBLIC_DEBUG_MODE,
        NODE_ENV: process.env.NODE_ENV,
        HAS_HF_TOKEN: !!process.env.NEXT_PUBLIC_HF_TOKEN,
      },
      apiStatus: {
        mode: config.mode,
        gradioEnabled: isGradioEnabled(),
        skipLocal: shouldSkipLocalAPI(),
      }
    };

    setDebugInfo(info);
  };

  if (!showPanel) {
    return null;
  }

  return (
    <div className="fixed bottom-4 left-4 z-50 max-w-md">
      {/* Toggle Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center space-x-2 px-3 py-2 bg-yellow-100 border border-yellow-300 rounded-lg shadow-sm hover:shadow-md transition-shadow text-sm font-medium text-yellow-800"
        title="Debug Panel (Debug Mode Only)"
      >
        <Bug className="h-4 w-4" />
        <span>Debug</span>
        {isOpen ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
      </button>

      {/* Debug Panel */}
      {isOpen && debugInfo && (
        <div className="mt-2 bg-white border border-gray-200 rounded-lg shadow-lg p-4 max-h-96 overflow-y-auto">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-gray-900">Debug Information</h3>
            <button
              onClick={() => {
                setRefreshCount(prev => prev + 1);
              }}
              className="text-xs text-blue-600 hover:text-blue-800 font-medium"
            >
              Refresh
            </button>
          </div>

          <div className="space-y-4 text-xs">
            {/* API Status */}
            <div>
              <h4 className="font-medium text-gray-700 mb-2">API Configuration</h4>
              <div className="bg-gray-50 rounded p-2 space-y-1">
                <div className="flex justify-between">
                  <span className="text-gray-600">Mode:</span>
                  <span className={`font-medium ${debugInfo.apiStatus.mode === 'gradio-only' ? 'text-blue-600' : 'text-green-600'
                    }`}>
                    {debugInfo.apiStatus.mode}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Gradio Enabled:</span>
                  <span className="flex items-center space-x-1">
                    {debugInfo.apiStatus.gradioEnabled ? (
                      <>
                        <CheckCircle className="h-3 w-3 text-green-500" />
                        <span className="text-green-600">Yes</span>
                      </>
                    ) : (
                      <>
                        <XCircle className="h-3 w-3 text-red-500" />
                        <span className="text-red-600">No</span>
                      </>
                    )}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Skip Local API:</span>
                  <span className="flex items-center space-x-1">
                    {debugInfo.apiStatus.skipLocal ? (
                      <>
                        <AlertTriangle className="h-3 w-3 text-yellow-500" />
                        <span className="text-yellow-600">Yes (CORS Safe)</span>
                      </>
                    ) : (
                      <>
                        <CheckCircle className="h-3 w-3 text-green-500" />
                        <span className="text-green-600">No</span>
                      </>
                    )}
                  </span>
                </div>
              </div>
            </div>

            {/* Environment Variables */}
            <div>
              <h4 className="font-medium text-gray-700 mb-2">Environment Variables</h4>
              <div className="bg-gray-900 rounded p-2 text-green-400 font-mono text-xs space-y-1">
                {Object.entries(debugInfo.environment).map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <span className="text-blue-300">{key}:</span>
                    <span className={value ? 'text-green-400' : 'text-red-400'}>
                      {String(value) || 'undefined'}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Runtime Config */}
            <div>
              <h4 className="font-medium text-gray-700 mb-2">Runtime Configuration</h4>
              <div className="bg-blue-50 rounded p-2 space-y-1">
                <div className="flex justify-between">
                  <span className="text-gray-600">Primary Service:</span>
                  <span className="font-medium flex items-center space-x-1">
                    {debugInfo.config.primaryService === 'gradio' ? (
                      <>
                        <Globe className="h-3 w-3 text-blue-500" />
                        <span className="text-blue-600">Gradio</span>
                      </>
                    ) : (
                      <>
                        <Server className="h-3 w-3 text-green-500" />
                        <span className="text-green-600">Local</span>
                      </>
                    )}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Will Use CORS:</span>
                  <span className={`font-medium ${debugInfo.config.willUseCORS ? 'text-yellow-600' : 'text-green-600'}`}>
                    {debugInfo.config.willUseCORS ? 'Yes' : 'No'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Debug Mode:</span>
                  <span className={`font-medium ${debugInfo.config.debugMode ? 'text-yellow-600' : 'text-gray-600'}`}>
                    {debugInfo.config.debugMode ? 'Enabled' : 'Disabled'}
                  </span>
                </div>
              </div>
            </div>

            {/* Recommendations */}
            <div>
              <h4 className="font-medium text-gray-700 mb-2">Status & Recommendations</h4>
              <div className="space-y-2">
                {debugInfo.apiStatus.gradioEnabled ? (
                  <div className="flex items-start space-x-2 p-2 bg-blue-50 rounded">
                    <CheckCircle className="h-4 w-4 text-blue-500 mt-0.5 flex-shrink-0" />
                    <div>
                      <div className="font-medium text-blue-700">Gradio Mode Active</div>
                      <div className="text-blue-600">
                        Using HuggingFace Space. Local API calls are skipped to prevent CORS errors.
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-start space-x-2 p-2 bg-green-50 rounded">
                    <CheckCircle className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                    <div>
                      <div className="font-medium text-green-700">Local API Mode</div>
                      <div className="text-green-600">
                        Using local server. Make sure it's running on the configured port.
                      </div>
                    </div>
                  </div>
                )}

                {debugInfo.config.debugMode && (
                  <div className="flex items-start space-x-2 p-2 bg-yellow-50 rounded">
                    <Bug className="h-4 w-4 text-yellow-500 mt-0.5 flex-shrink-0" />
                    <div>
                      <div className="font-medium text-yellow-700">Debug Mode Enabled</div>
                      <div className="text-yellow-600">
                        Check browser console for detailed API logs.
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Timestamp */}
            <div className="pt-2 border-t border-gray-200 text-center text-gray-500">
              Last updated: {debugInfo.timestamp}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
