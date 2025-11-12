'use client';

import React, { useState, useEffect } from 'react';
import { Settings, Globe, Server, CheckCircle, XCircle, AlertCircle, Loader2 } from 'lucide-react';
import { getGradioConfig, isGradioEnabled, gradioClient } from '../lib/gradio-client';
import { getHealth, getModelInfo, getConfigSummary } from '../lib/api-wrapper';

interface ApiStatus {
  type: 'local' | 'gradio';
  status: 'connected' | 'error' | 'checking';
  message?: string;
  details?: any;
}

export default function ApiSettings() {
  const [isOpen, setIsOpen] = useState(false);
  const [apiStatus, setApiStatus] = useState<{
    local: ApiStatus;
    gradio: ApiStatus;
  }>({
    local: { type: 'local', status: 'checking' },
    gradio: { type: 'gradio', status: 'checking' }
  });

  const gradioConfig = getGradioConfig();
  const isGradioActive = isGradioEnabled();

  // Check API status
  useEffect(() => {
    // Log configuration
    const config = getConfigSummary();
    console.log('⚙️ API Settings Configuration:', config);

    checkApiStatus();
  }, []);

  const checkApiStatus = async () => {
    // Check local API
    try {
      const health = await getHealth();
      setApiStatus(prev => ({
        ...prev,
        local: {
          type: 'local',
          status: health.status === 'healthy' ? 'connected' : 'error',
          message: health.status === 'healthy' ? 'API is running' : 'API not healthy',
          details: health
        }
      }));
    } catch (error) {
      setApiStatus(prev => ({
        ...prev,
        local: {
          type: 'local',
          status: 'error',
          message: 'Failed to connect to local API',
          details: error
        }
      }));
    }

    // Check Gradio if enabled
    if (isGradioActive) {
      try {
        const isConnected = await gradioClient.testConnection();
        const spaceInfo = await gradioClient.getSpaceInfo();

        setApiStatus(prev => ({
          ...prev,
          gradio: {
            type: 'gradio',
            status: isConnected ? 'connected' : 'error',
            message: isConnected ? 'Gradio space connected' : 'Failed to connect to Gradio space',
            details: spaceInfo
          }
        }));
      } catch (error) {
        setApiStatus(prev => ({
          ...prev,
          gradio: {
            type: 'gradio',
            status: 'error',
            message: 'Gradio connection failed',
            details: error
          }
        }));
      }
    } else {
      setApiStatus(prev => ({
        ...prev,
        gradio: {
          type: 'gradio',
          status: 'error',
          message: 'Gradio is disabled',
          details: { reason: 'NEXT_PUBLIC_USE_GRADIO not set to true' }
        }
      }));
    }
  };

  const getStatusIcon = (status: ApiStatus['status']) => {
    switch (status) {
      case 'connected':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'error':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'checking':
        return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />;
      default:
        return <AlertCircle className="h-4 w-4 text-yellow-500" />;
    }
  };

  const getStatusColor = (status: ApiStatus['status']) => {
    switch (status) {
      case 'connected':
        return 'bg-green-50 border-green-200';
      case 'error':
        return 'bg-red-50 border-red-200';
      case 'checking':
        return 'bg-blue-50 border-blue-200';
      default:
        return 'bg-gray-50 border-gray-200';
    }
  };

  return (
    <>
      {/* Settings Button */}
      <button
        onClick={() => setIsOpen(true)}
        className="fixed top-4 right-4 z-40 p-2 bg-white border border-gray-200 rounded-lg shadow-sm hover:shadow-md transition-shadow"
        title="API Settings"
      >
        <Settings className="h-5 w-5 text-gray-600" />
      </button>

      {/* Settings Modal */}
      {isOpen && (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          <div className="flex min-h-screen items-center justify-center p-4 text-center sm:p-0">
            {/* Backdrop */}
            <div
              className="fixed inset-0 bg-black bg-opacity-25 transition-opacity"
              onClick={() => setIsOpen(false)}
            />

            {/* Modal Content */}
            <div className="relative transform overflow-hidden rounded-lg bg-white px-4 pb-4 pt-5 text-left shadow-xl transition-all sm:my-8 sm:w-full sm:max-w-2xl sm:p-6">
              <div className="absolute right-0 top-0 pr-4 pt-4">
                <button
                  type="button"
                  className="rounded-md bg-white text-gray-400 hover:text-gray-500 focus:outline-none"
                  onClick={() => setIsOpen(false)}
                >
                  <span className="sr-only">Close</span>
                  <XCircle className="h-6 w-6" />
                </button>
              </div>

              <div className="sm:flex sm:items-start">
                <div className="mt-3 text-center sm:ml-4 sm:mt-0 sm:text-left w-full">
                  <h3 className="text-lg font-medium leading-6 text-gray-900 mb-4">
                    API Configuration & Status
                  </h3>

                  {/* Current Configuration */}
                  <div className="mb-6">
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Current Configuration</h4>
                    <div className="bg-gray-50 rounded-lg p-4 space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Active API:</span>
                        <span className="font-medium">
                          {isGradioActive ? 'Gradio Only (No Local Fallback)' : 'Local API Server'}
                        </span>
                      </div>
                      {isGradioActive && (
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600">Space ID:</span>
                          <span className="font-mono text-xs">{gradioConfig.spaceId}</span>
                        </div>
                      )}
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Debug Mode:</span>
                        <span className="font-medium">
                          {process.env.NEXT_PUBLIC_DEBUG_MODE === 'true' ? 'Enabled' : 'Disabled'}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* API Status */}
                  <div className="mb-6">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="text-sm font-medium text-gray-700">API Status</h4>
                      <button
                        onClick={checkApiStatus}
                        className="text-xs text-blue-600 hover:text-blue-800 font-medium"
                      >
                        Refresh Status
                      </button>
                    </div>

                    <div className="space-y-3">
                      {/* Local API Status */}
                      <div className={`p-3 rounded-lg border ${getStatusColor(apiStatus.local.status)}`}>
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-3">
                            <Server className="h-5 w-5 text-gray-600" />
                            <div>
                              <div className="text-sm font-medium text-gray-900">Local API Server</div>
                              <div className="text-xs text-gray-600">
                                {process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'}
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center space-x-2">
                            {getStatusIcon(apiStatus.local.status)}
                            <span className="text-xs font-medium">
                              {apiStatus.local.message}
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* Gradio Status */}
                      <div className={`p-3 rounded-lg border ${getStatusColor(apiStatus.gradio.status)}`}>
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-3">
                            <Globe className="h-5 w-5 text-gray-600" />
                            <div>
                              <div className="text-sm font-medium text-gray-900">Gradio HuggingFace Space</div>
                              <div className="text-xs text-gray-600">
                                {gradioConfig.spaceId}
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center space-x-2">
                            {getStatusIcon(apiStatus.gradio.status)}
                            <span className="text-xs font-medium">
                              {apiStatus.gradio.message}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Environment Variables Guide */}
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Environment Variables</h4>
                    <div className="bg-gray-900 rounded-lg p-4 text-xs text-green-400 font-mono overflow-x-auto">
                      <div className="space-y-1">
                        <div># Enable Gradio (HuggingFace Space)</div>
                        <div>NEXT_PUBLIC_USE_GRADIO=true</div>
                        <div>NEXT_PUBLIC_GRADIO_SPACE_ID=TwiWhisperModel/TwiSpeechModel</div>
                        <div>NEXT_PUBLIC_HF_TOKEN=hf_your_token_here</div>
                        <div className="mt-3"># Use Local API</div>
                        <div>NEXT_PUBLIC_USE_GRADIO=false</div>
                        <div>NEXT_PUBLIC_API_BASE_URL=http://localhost:8000</div>
                        <div className="mt-3"># Debug</div>
                        <div>NEXT_PUBLIC_DEBUG_MODE=true</div>
                      </div>
                    </div>
                  </div>

                  {/* Usage Info */}
                  <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                    <div className="text-xs text-blue-700">
                      <strong>Note:</strong> When Gradio is enabled, the app uses only Gradio to avoid CORS errors.
                      Local API calls are skipped entirely when NEXT_PUBLIC_USE_GRADIO=true.
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
