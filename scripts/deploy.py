#!/usr/bin/env python3
"""
Automated Deployment Script

This script provides automated deployment capabilities for the Twi Speech Model,
supporting various deployment targets including local, Docker, cloud platforms,
and CI/CD pipelines.

Features:
- Multi-platform deployment (local, Docker, AWS, GCP, Azure)
- Environment management and configuration
- Health checking and monitoring setup
- Rollback capabilities
- Load balancing and scaling
- SSL/TLS configuration
- Logging and metrics collection
"""

import os
import sys
import json
import yaml
import argparse
import subprocess
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import tempfile
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelDeployer:
    """Professional deployment automation system."""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "deployment_config.yaml"
        self.config = self._load_config()
        self.deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration."""
        default_config = {
            "app": {
                "name": "twi-speech-model",
                "version": "1.0.0",
                "port": 8000,
                "workers": 4,
                "timeout": 30,
                "max_requests": 1000,
                "max_requests_jitter": 50
            },
            "environments": {
                "development": {
                    "domain": "localhost",
                    "ssl": False,
                    "debug": True,
                    "log_level": "DEBUG"
                },
                "staging": {
                    "domain": "staging.example.com",
                    "ssl": True,
                    "debug": False,
                    "log_level": "INFO"
                },
                "production": {
                    "domain": "api.example.com",
                    "ssl": True,
                    "debug": False,
                    "log_level": "WARNING",
                    "replicas": 3,
                    "health_check_interval": 30
                }
            },
            "docker": {
                "image_name": "twi-speech-model",
                "registry": "docker.io",
                "dockerfile": "Dockerfile",
                "build_args": {},
                "ports": {"8000/tcp": 8000},
                "volumes": {},
                "environment": {}
            },
            "kubernetes": {
                "namespace": "default",
                "deployment_name": "twi-speech-model",
                "service_name": "twi-speech-model-service",
                "ingress_name": "twi-speech-model-ingress",
                "config_map": "twi-speech-model-config",
                "secret": "twi-speech-model-secret"
            },
            "aws": {
                "region": "us-west-2",
                "ecs_cluster": "ml-models",
                "task_definition": "twi-speech-model",
                "service_name": "twi-speech-model-service",
                "load_balancer": "twi-speech-model-alb",
                "target_group": "twi-speech-model-tg"
            },
            "monitoring": {
                "health_endpoint": "/health",
                "metrics_endpoint": "/metrics",
                "log_retention_days": 30,
                "alert_email": "alerts@example.com"
            }
        }

        config_path = Path(self.config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    if config_path.suffix.lower() in ['.yml', '.yaml']:
                        user_config = yaml.safe_load(f)
                    else:
                        user_config = json.load(f)

                # Deep merge configurations
                return self._deep_merge(default_config, user_config)
            except Exception as e:
                logger.warning(f"Error loading config file {config_path}: {e}")
                logger.info("Using default configuration")

        return default_config

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def validate_environment(self, environment: str) -> bool:
        """Validate deployment environment."""
        logger.info(f"Validating environment: {environment}")

        if environment not in self.config["environments"]:
            logger.error(f"Unknown environment: {environment}")
            return False

        env_config = self.config["environments"][environment]

        # Check required tools
        required_tools = ["docker"]
        for tool in required_tools:
            if not self._check_command_exists(tool):
                logger.error(f"Required tool not found: {tool}")
                return False

        # Check Docker daemon
        try:
            subprocess.run(["docker", "info"], check=True, capture_output=True)
            logger.info("✓ Docker daemon is running")
        except subprocess.CalledProcessError:
            logger.error("✗ Docker daemon is not running")
            return False

        logger.info("✓ Environment validation passed")
        return True

    def _check_command_exists(self, command: str) -> bool:
        """Check if a command exists in PATH."""
        return shutil.which(command) is not None

    def build_docker_image(self, tag: Optional[str] = None) -> str:
        """Build Docker image for the model."""
        logger.info("Building Docker image...")

        if tag is None:
            tag = f"{self.config['docker']['image_name']}:{self.config['app']['version']}"

        dockerfile = self.config['docker']['dockerfile']
        build_args = self.config['docker']['build_args']

        # Prepare build command
        cmd = ["docker", "build", "-t", tag, "-f", dockerfile]

        # Add build arguments
        for key, value in build_args.items():
            cmd.extend(["--build-arg", f"{key}={value}"])

        cmd.append(".")

        try:
            logger.info(f"Building image with tag: {tag}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("✓ Docker image built successfully")
            logger.debug(f"Build output: {result.stdout}")
            return tag
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Docker build failed: {e}")
            logger.error(f"Build error: {e.stderr}")
            raise

    def deploy_local(self, environment: str, image_tag: str) -> bool:
        """Deploy model locally using Docker."""
        logger.info(f"Deploying locally to {environment} environment...")

        env_config = self.config["environments"][environment]
        docker_config = self.config["docker"]
        app_config = self.config["app"]

        container_name = f"{app_config['name']}-{environment}"

        # Stop existing container if running
        try:
            subprocess.run(["docker", "stop", container_name], check=True, capture_output=True)
            subprocess.run(["docker", "rm", container_name], check=True, capture_output=True)
            logger.info(f"Stopped existing container: {container_name}")
        except subprocess.CalledProcessError:
            pass  # Container might not exist

        # Prepare run command
        cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "--restart", "unless-stopped"
        ]

        # Add port mappings
        for container_port, host_port in docker_config["ports"].items():
            cmd.extend(["-p", f"{host_port}:{container_port.split('/')[0]}"])

        # Add volume mappings
        for host_path, container_path in docker_config["volumes"].items():
            cmd.extend(["-v", f"{host_path}:{container_path}"])

        # Add environment variables
        for key, value in docker_config["environment"].items():
            cmd.extend(["-e", f"{key}={value}"])

        # Add environment-specific variables
        cmd.extend(["-e", f"ENVIRONMENT={environment}"])
        cmd.extend(["-e", f"LOG_LEVEL={env_config.get('log_level', 'INFO')}"])
        cmd.extend(["-e", f"DEBUG={env_config.get('debug', False)}"])

        # Add image tag
        cmd.append(image_tag)

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            container_id = result.stdout.strip()
            logger.info(f"✓ Container started: {container_id[:12]}")

            # Wait for container to be ready
            if self._wait_for_health_check(environment):
                logger.info("✓ Local deployment successful")
                return True
            else:
                logger.error("✗ Health check failed")
                return False

        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Container start failed: {e}")
            logger.error(f"Error: {e.stderr}")
            return False

    def deploy_kubernetes(self, environment: str, image_tag: str) -> bool:
        """Deploy model to Kubernetes cluster."""
        logger.info(f"Deploying to Kubernetes ({environment})...")

        if not self._check_command_exists("kubectl"):
            logger.error("kubectl not found. Please install Kubernetes CLI")
            return False

        # Generate Kubernetes manifests
        manifests_dir = self._generate_k8s_manifests(environment, image_tag)

        try:
            # Apply manifests
            cmd = ["kubectl", "apply", "-f", str(manifests_dir)]
            subprocess.run(cmd, check=True, capture_output=True)

            # Wait for deployment to be ready
            deployment_name = self.config["kubernetes"]["deployment_name"]
            namespace = self.config["kubernetes"]["namespace"]

            cmd = [
                "kubectl", "rollout", "status",
                f"deployment/{deployment_name}",
                "-n", namespace,
                "--timeout=300s"
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            logger.info("✓ Kubernetes deployment successful")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Kubernetes deployment failed: {e}")
            return False
        finally:
            # Clean up temporary manifests
            shutil.rmtree(manifests_dir, ignore_errors=True)

    def _generate_k8s_manifests(self, environment: str, image_tag: str) -> Path:
        """Generate Kubernetes deployment manifests."""
        temp_dir = Path(tempfile.mkdtemp(prefix="k8s-manifests-"))

        env_config = self.config["environments"][environment]
        k8s_config = self.config["kubernetes"]
        app_config = self.config["app"]

        # Deployment manifest
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": k8s_config["deployment_name"],
                "namespace": k8s_config["namespace"],
                "labels": {
                    "app": app_config["name"],
                    "environment": environment,
                    "version": app_config["version"]
                }
            },
            "spec": {
                "replicas": env_config.get("replicas", 1),
                "selector": {
                    "matchLabels": {
                        "app": app_config["name"],
                        "environment": environment
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": app_config["name"],
                            "environment": environment,
                            "version": app_config["version"]
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": app_config["name"],
                            "image": image_tag,
                            "ports": [{
                                "containerPort": app_config["port"],
                                "name": "http"
                            }],
                            "env": [
                                {"name": "ENVIRONMENT", "value": environment},
                                {"name": "LOG_LEVEL", "value": env_config.get("log_level", "INFO")},
                                {"name": "DEBUG", "value": str(env_config.get("debug", False))}
                            ],
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": "http"
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": "http"
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            },
                            "resources": {
                                "requests": {
                                    "cpu": "500m",
                                    "memory": "1Gi"
                                },
                                "limits": {
                                    "cpu": "2000m",
                                    "memory": "4Gi"
                                }
                            }
                        }]
                    }
                }
            }
        }

        # Service manifest
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": k8s_config["service_name"],
                "namespace": k8s_config["namespace"],
                "labels": {
                    "app": app_config["name"],
                    "environment": environment
                }
            },
            "spec": {
                "selector": {
                    "app": app_config["name"],
                    "environment": environment
                },
                "ports": [{
                    "port": 80,
                    "targetPort": "http",
                    "protocol": "TCP"
                }],
                "type": "ClusterIP"
            }
        }

        # Ingress manifest (if SSL is enabled)
        if env_config.get("ssl", False):
            ingress = {
                "apiVersion": "networking.k8s.io/v1",
                "kind": "Ingress",
                "metadata": {
                    "name": k8s_config["ingress_name"],
                    "namespace": k8s_config["namespace"],
                    "annotations": {
                        "kubernetes.io/ingress.class": "nginx",
                        "cert-manager.io/cluster-issuer": "letsencrypt-prod"
                    }
                },
                "spec": {
                    "tls": [{
                        "hosts": [env_config["domain"]],
                        "secretName": f"{app_config['name']}-tls"
                    }],
                    "rules": [{
                        "host": env_config["domain"],
                        "http": {
                            "paths": [{
                                "path": "/",
                                "pathType": "Prefix",
                                "backend": {
                                    "service": {
                                        "name": k8s_config["service_name"],
                                        "port": {"number": 80}
                                    }
                                }
                            }]
                        }
                    }]
                }
            }

        # Write manifests to files
        with open(temp_dir / "deployment.yaml", 'w') as f:
            yaml.dump(deployment, f, default_flow_style=False)

        with open(temp_dir / "service.yaml", 'w') as f:
            yaml.dump(service, f, default_flow_style=False)

        if env_config.get("ssl", False):
            with open(temp_dir / "ingress.yaml", 'w') as f:
                yaml.dump(ingress, f, default_flow_style=False)

        return temp_dir

    def deploy_aws_ecs(self, environment: str, image_tag: str) -> bool:
        """Deploy model to AWS ECS."""
        logger.info(f"Deploying to AWS ECS ({environment})...")

        if not self._check_command_exists("aws"):
            logger.error("AWS CLI not found. Please install AWS CLI")
            return False

        # Push image to ECR
        ecr_uri = self._push_to_ecr(image_tag)
        if not ecr_uri:
            return False

        # Update ECS service
        return self._update_ecs_service(environment, ecr_uri)

    def _push_to_ecr(self, image_tag: str) -> Optional[str]:
        """Push Docker image to AWS ECR."""
        try:
            # Get ECR login token
            cmd = ["aws", "ecr", "get-login-password", "--region", self.config["aws"]["region"]]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            login_token = result.stdout.strip()

            # Get ECR repository URI
            repository_name = self.config["docker"]["image_name"]
            cmd = [
                "aws", "ecr", "describe-repositories",
                "--repository-names", repository_name,
                "--region", self.config["aws"]["region"]
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            repo_data = json.loads(result.stdout)
            ecr_uri = repo_data["repositories"][0]["repositoryUri"]

            # Tag image for ECR
            ecr_tag = f"{ecr_uri}:{self.config['app']['version']}"
            subprocess.run(["docker", "tag", image_tag, ecr_tag], check=True)

            # Login to ECR
            registry_url = ecr_uri.split('/')[0]
            cmd = f"echo {login_token} | docker login --username AWS --password-stdin {registry_url}"
            subprocess.run(cmd, shell=True, check=True)

            # Push image
            subprocess.run(["docker", "push", ecr_tag], check=True)

            logger.info(f"✓ Image pushed to ECR: {ecr_tag}")
            return ecr_tag

        except subprocess.CalledProcessError as e:
            logger.error(f"✗ ECR push failed: {e}")
            return None

    def _update_ecs_service(self, environment: str, image_uri: str) -> bool:
        """Update ECS service with new image."""
        try:
            aws_config = self.config["aws"]

            # Update task definition
            # This is a simplified version - in practice, you'd need to
            # fetch the current task definition and update it
            task_def_update = {
                "family": aws_config["task_definition"],
                "containerDefinitions": [{
                    "name": self.config["app"]["name"],
                    "image": image_uri,
                    "portMappings": [{
                        "containerPort": self.config["app"]["port"],
                        "protocol": "tcp"
                    }],
                    "environment": [
                        {"name": "ENVIRONMENT", "value": environment}
                    ]
                }]
            }

            # Register new task definition
            cmd = [
                "aws", "ecs", "register-task-definition",
                "--cli-input-json", json.dumps(task_def_update),
                "--region", aws_config["region"]
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            # Update service
            cmd = [
                "aws", "ecs", "update-service",
                "--cluster", aws_config["ecs_cluster"],
                "--service", aws_config["service_name"],
                "--task-definition", aws_config["task_definition"],
                "--region", aws_config["region"]
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            logger.info("✓ ECS service updated")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"✗ ECS update failed: {e}")
            return False

    def _wait_for_health_check(self, environment: str, timeout: int = 300) -> bool:
        """Wait for service to pass health checks."""
        env_config = self.config["environments"][environment]
        health_endpoint = self.config["monitoring"]["health_endpoint"]

        if environment == "local" or environment == "development":
            base_url = f"http://{env_config['domain']}:{self.config['app']['port']}"
        else:
            protocol = "https" if env_config.get("ssl", False) else "http"
            base_url = f"{protocol}://{env_config['domain']}"

        health_url = f"{base_url}{health_endpoint}"

        logger.info(f"Waiting for health check at {health_url}")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=10)
                if response.status_code == 200:
                    logger.info("✓ Health check passed")
                    return True
            except requests.RequestException:
                pass

            time.sleep(5)

        logger.error("✗ Health check timeout")
        return False

    def rollback(self, environment: str, target_version: Optional[str] = None) -> bool:
        """Rollback to previous deployment."""
        logger.info(f"Rolling back deployment in {environment}")

        # Implementation would depend on deployment target
        # For Kubernetes, this would use kubectl rollout undo
        # For ECS, this would revert to previous task definition
        # For local Docker, this would start the previous image

        try:
            if self._check_command_exists("kubectl"):
                # Kubernetes rollback
                k8s_config = self.config["kubernetes"]
                cmd = [
                    "kubectl", "rollout", "undo",
                    f"deployment/{k8s_config['deployment_name']}",
                    "-n", k8s_config["namespace"]
                ]
                if target_version:
                    cmd.extend(["--to-revision", target_version])

                subprocess.run(cmd, check=True, capture_output=True)
                logger.info("✓ Kubernetes rollback successful")
                return True

        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Rollback failed: {e}")
            return False

        return False

    def get_deployment_status(self, environment: str) -> Dict[str, Any]:
        """Get current deployment status."""
        status = {
            "environment": environment,
            "healthy": False,
            "version": "unknown",
            "replicas": 0,
            "ready_replicas": 0,
            "timestamp": datetime.now().isoformat()
        }

        try:
            # Check health endpoint
            env_config = self.config["environments"][environment]
            health_endpoint = self.config["monitoring"]["health_endpoint"]

            if environment == "local" or environment == "development":
                base_url = f"http://{env_config['domain']}:{self.config['app']['port']}"
            else:
                protocol = "https" if env_config.get("ssl", False) else "http"
                base_url = f"{protocol}://{env_config['domain']}"

            health_url = f"{base_url}{health_endpoint}"

            response = requests.get(health_url, timeout=10)
            if response.status_code == 200:
                status["healthy"] = True
                health_data = response.json()
                status["version"] = health_data.get("version", "unknown")

            # Get Kubernetes status if available
            if self._check_command_exists("kubectl"):
                k8s_config = self.config["kubernetes"]
                cmd = [
                    "kubectl", "get", "deployment",
                    k8s_config["deployment_name"],
                    "-n", k8s_config["namespace"],
                    "-o", "json"
                ]
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                deployment_data = json.loads(result.stdout)

                status["replicas"] = deployment_data["spec"]["replicas"]
                status["ready_replicas"] = deployment_data["status"].get("readyReplicas", 0)

        except Exception as e:
            logger.warning(f"Could not get full deployment status: {e}")

        return status

    def generate_deployment_report(self, environment: str) -> str:
        """Generate deployment report."""
        status = self.get_deployment_status(environment)

        report = f"""
Deployment Report - {environment.upper()}
{'='*50}
Environment: {status['environment']}
Status: {'✓ HEALTHY' if status['healthy'] else '✗ UNHEALTHY'}
Version: {status['version']}
Replicas: {status['ready_replicas']}/{status['replicas']}
Timestamp: {status['timestamp']}
{'='*50}
"""
        return report


def main():
    """Main entry point for the deployment script."""
    parser = argparse.ArgumentParser(
        description="Automated Model Deployment System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "action",
        choices=["deploy", "rollback", "status", "build"],
        help="Deployment action to perform"
    )

    parser.add_argument(
        "--environment", "-e",
        required=True,
        help="Target environment (development, staging, production)"
    )

    parser.add_argument(
        "--target", "-t",
        choices=["local", "kubernetes", "aws-ecs"],
        default="local",
        help="Deployment target"
    )

    parser.add_argument(
        "--config", "-c",
        help="Path to deployment configuration file"
    )

    parser.add_argument(
        "--image-tag",
        help="Docker image tag to deploy"
    )

    parser.add_argument(
        "--version",
        help="Version to rollback to (for rollback action)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        deployer = ModelDeployer(args.config)

        if args.action == "build":
            # Build Docker image
            if not deployer.validate_environment(args.environment):
                sys.exit(1)

            image_tag = deployer.build_docker_image(args.image_tag)
            print(f"✓ Built image: {image_tag}")

        elif args.action == "deploy":
            # Deploy model
            if not deployer.validate_environment(args.environment):
                sys.exit(1)

            # Build image if not provided
            image_tag = args.image_tag
            if not image_tag:
                image_tag = deployer.build_docker_image()

            # Deploy based on target
            success = False
            if args.target == "local":
                success = deployer.deploy_local(args.environment, image_tag)
            elif args.target == "kubernetes":
                success = deployer.deploy_kubernetes(args.environment, image_tag)
            elif args.target == "aws-ecs":
                success = deployer.deploy_aws_ecs(args.environment, image_tag)

            if success:
                print(deployer.generate_deployment_report(args.environment))
            else:
                sys.exit(1)

        elif args.action == "rollback":
            # Rollback deployment
            success = deployer.rollback(args.environment, args.version)
            if not success:
                sys.exit(1)

        elif args.action == "status":
            # Get deployment status
            print(deployer.generate_deployment_report(args.environment))

    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
