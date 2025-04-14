#!/usr/bin/env python3
"""
OpenShift AI Agent with LLM Integration
This agent combines OpenShift management with LLM capabilities to:
- Understand natural language commands
- Execute OpenShift operations (start/stop pods, HPA)
- Provide intelligent responses and recommendations
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
import requests
from requests.auth import HTTPBasicAuth
from openai import OpenAI  # or any other LLM provider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("OpenShiftAIAgent")

class OpenShiftAIAgent:
    def __init__(self, config: Dict):
        """
        Initialize the AI agent with OpenShift and LLM configuration
        
        Args:
            config: Dictionary containing:
                   - openshift_url, username, password, verify_ssl
                   - llm_api_key, llm_model (optional), llm_provider
        """
        # OpenShift configuration
        self.openshift_url = config['openshift_url'].rstrip('/')
        self.auth = HTTPBasicAuth(config['username'], config['password'])
        self.verify_ssl = config.get('verify_ssl', True)
        self.default_namespace = config.get('default_namespace', 'default')
        
        # LLM configuration
        self.llm_provider = config.get('llm_provider', 'openai')
        self.llm_model = config.get('llm_model', 'gpt-4')
        self.llm_client = self._initialize_llm(config['llm_api_key'])
        
        # System prompt for LLM
        self.system_prompt = """
        You are an expert OpenShift administrator AI assistant. Your capabilities include:
        - Managing OpenShift pods (starting, stopping, scaling)
        - Configuring Horizontal Pod Autoscalers (HPA)
        - Analyzing cluster metrics and making recommendations
        - Explaining OpenShift concepts in simple terms
        
        You will receive requests from users and need to:
        1. Determine if the request is a direct command or requires analysis
        2. For commands: identify the exact operation needed
        3. For analysis: provide insights based on available data
        
        Always be concise but thorough in your responses.
        """
        
        # Test connections
        self._test_connections()

    def _initialize_llm(self, api_key: str):
        """Initialize the LLM client based on provider"""
        if self.llm_provider == 'openai':
            return OpenAI(api_key=api_key)
        elif self.llm_provider == 'anthropic':
            # Example for Anthropic Claude
            from anthropic import Anthropic
            return Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _test_connections(self):
        """Test connections to both OpenShift and LLM service"""
        # Test OpenShift connection
        try:
            response = requests.get(
                f"{self.openshift_url}/apis",
                auth=self.auth,
                verify=self.verify_ssl
            )
            response.raise_for_status()
            logger.info("Successfully connected to OpenShift cluster")
        except Exception as e:
            logger.error(f"Failed to connect to OpenShift cluster: {e}")
            raise

        # Test LLM connection
        try:
            if self.llm_provider == 'openai':
                self.llm_client.models.list()
            logger.info("Successfully connected to LLM service")
        except Exception as e:
            logger.error(f"Failed to connect to LLM service: {e}")
            raise

    def _make_openshift_request(self, method: str, path: str, data: Optional[Dict] = None) -> Dict:
        """Make an API request to OpenShift"""
        url = f"{self.openshift_url}{path}"
        headers = {'Content-Type': 'application/json'}
        
        try:
            response = requests.request(
                method,
                url,
                auth=self.auth,
                headers=headers,
                json=data,
                verify=self.verify_ssl
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"OpenShift API request failed: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise

    def _query_llm(self, prompt: str, context: str = "") -> str:
        """Query the LLM with a prompt and optional context"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": context + "\n\n" + prompt if context else prompt}
        ]
        
        try:
            if self.llm_provider == 'openai':
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    temperature=0.3  # Lower for more deterministic responses
                )
                return response.choices[0].message.content
            elif self.llm_provider == 'anthropic':
                response = self.llm_client.messages.create(
                    model=self.llm_model,
                    messages=messages,
                    max_tokens=1000
                )
                return response.content[0].text
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            raise

    def process_command(self, natural_language_command: str) -> Tuple[str, Dict]:
        """
        Process a natural language command and return both human-readable response
        and structured data about the operation to perform
        
        Returns:
            Tuple of (human_response, operation_data)
        """
        prompt = f"""
        The user provided this command: {natural_language_command}
        
        Analyze this command and:
        1. Determine if it's a valid OpenShift operation request
        2. If valid, return the operation details in JSON format
        3. If not valid, explain why and suggest alternatives
        
        For valid operations, respond with JSON in this format:
        {{
            "operation": "<operation_type>",
            "target": "<resource_name>",
            "parameters": {{
                "<param1>": "<value1>",
                "<param2>": "<value2>"
            }},
            "namespace": "<namespace_name>",
            "confirmation_required": true/false
        }}
        
        Include your human-readable response before the JSON.
        """
        
        response = self._query_llm(prompt)
        
        # Extract JSON from response
        try:
            json_start = response.rfind('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            operation_data = json.loads(json_str)
            human_response = response[:json_start].strip()
        except (ValueError, json.JSONDecodeError) as e:
            human_response = response
            operation_data = None
            logger.warning(f"Could not parse operation data from LLM response: {e}")
        
        return human_response, operation_data

    def execute_operation(self, operation_data: Dict) -> str:
        """Execute an OpenShift operation based on structured data"""
        if not operation_data:
            return "No valid operation to execute"
        
        try:
            op_type = operation_data['operation'].lower()
            target = operation_data['target']
            namespace = operation_data.get('namespace', self.default_namespace)
            params = operation_data.get('parameters', {})
            
            if op_type == 'scale_deployment':
                replicas = params.get('replicas', 1)
                result = self._scale_deployment(target, replicas, namespace)
                return f"Successfully scaled deployment {target} to {replicas} replicas"
            
            elif op_type == 'start_pods':
                replicas = params.get('replicas', 1)
                result = self._scale_deployment(target, replicas, namespace)
                return f"Successfully started {replicas} pods for deployment {target}"
            
            elif op_type == 'stop_pods':
                result = self._scale_deployment(target, 0, namespace)
                return f"Successfully stopped pods for deployment {target}"
            
            elif op_type == 'create_hpa':
                hpa_config = {
                    "apiVersion": "autoscaling/v1",
                    "kind": "HorizontalPodAutoscaler",
                    "metadata": {"name": target},
                    "spec": {
                        "scaleTargetRef": {
                            "apiVersion": "apps/v1",
                            "kind": "Deployment",
                            "name": params.get('deployment', target)
                        },
                        "minReplicas": params.get('min_replicas', 1),
                        "maxReplicas": params.get('max_replicas', 10),
                        "targetCPUUtilizationPercentage": params.get('cpu_target', 80)
                    }
                }
                result = self._make_openshift_request('POST', 
                    f"/apis/autoscaling/v1/namespaces/{namespace}/horizontalpodautoscalers",
                    hpa_config)
                return f"Successfully created HPA {target}"
            
            else:
                return f"Unsupported operation type: {op_type}"
                
        except Exception as e:
            return f"Operation failed: {str(e)}"

    def _scale_deployment(self, deployment: str, replicas: int, namespace: str) -> Dict:
        """Scale a deployment to a specific number of replicas"""
        path = f"/apis/apps/v1/namespaces/{namespace}/deployments/{deployment}/scale"
        
        # Current scale
        current = self._make_openshift_request('GET', path)
        
        # Update scale
        patch = {
            "kind": "Scale",
            "apiVersion": "autoscaling/v1",
            "metadata": current['metadata'],
            "spec": {"replicas": replicas}
        }
        
        return self._make_openshift_request('PATCH', path, patch)

    def analyze_cluster(self, question: str) -> str:
        """
        Analyze cluster state and provide insights using LLM
        
        Args:
            question: Natural language question about cluster state
            
        Returns:
            Human-readable analysis
        """
        # First gather relevant cluster data
        cluster_data = self._gather_cluster_data()
        
        prompt = f"""
        Cluster State:
        {json.dumps(cluster_data, indent=2)}
        
        Question: {question}
        
        Please analyze the cluster state and provide:
        1. Direct answer to the question
        2. Relevant metrics supporting your answer
        3. Any recommendations for optimization
        """
        
        return self._query_llm(prompt)

    def _gather_cluster_data(self) -> Dict:
        """Gather basic cluster data for analysis"""
        try:
            # Get cluster nodes
            nodes = self._make_openshift_request('GET', '/api/v1/nodes')
            node_count = len(nodes['items'])
            
            # Get namespaces
            namespaces = self._make_openshift_request('GET', '/api/v1/namespaces')
            namespace_count = len(namespaces['items'])
            
            # Get deployments in default namespace
            deployments = self._make_openshift_request('GET', 
                f"/apis/apps/v1/namespaces/{self.default_namespace}/deployments")
            deployment_count = len(deployments['items'])
            
            return {
                "node_count": node_count,
                "namespace_count": namespace_count,
                "deployment_count": deployment_count,
                "default_namespace": self.default_namespace,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            logger.error(f"Failed to gather cluster data: {e}")
            return {"error": str(e)}

    def interactive_chat(self, user_input: str) -> str:
        """
        Handle interactive chat with the agent
        
        Args:
            user_input: Natural language input from user
            
        Returns:
            Agent's response
        """
        # First determine if this is a command or question
        prompt = f"""
        User input: {user_input}
        
        Is this input:
        1. A direct command to perform an OpenShift operation?
        2. A question about cluster state or configuration?
        3. A general question about OpenShift?
        
        Respond with just the number (1, 2, or 3).
        """
        
        input_type = self._query_llm(prompt).strip()
        
        if input_type == '1':
            # Process as command
            human_response, operation_data = self.process_command(user_input)
            if operation_data:
                if operation_data.get('confirmation_required', False):
                    return f"{human_response}\n\nPlease confirm you want to execute this operation."
                else:
                    execution_result = self.execute_operation(operation_data)
                    return f"{human_response}\n\n{execution_result}"
            else:
                return human_response
        elif input_type == '2':
            # Analyze cluster state
            return self.analyze_cluster(user_input)
        else:
            # General OpenShift question
            return self._query_llm(user_input)

# Example usage
if __name__ == "__main__":
    # Configuration - replace with your actual credentials
    config = {
        # OpenShift configuration
        "openshift_url": "https://openshift.example.com:6443",
        "username": "admin",
        "password": "password",
        "verify_ssl": False,
        "default_namespace": "my-project",
        
        # LLM configuration
        "llm_provider": "openai",
        "llm_api_key": "your-api-key-here",
        "llm_model": "gpt-4"
    }
    
    # Initialize agent
    agent = OpenShiftAIAgent(config)
    
    # Example interaction loop
    print("OpenShift AI Agent - Type 'exit' to quit")
    while True:
        user_input = input("\n> ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        try:
            response = agent.interactive_chat(user_input)
            print("\n" + response)
        except Exception as e:
            print(f"Error: {str(e)}")
