"""
Bedrock enhancer - Enhance solutions using AWS Bedrock
"""
import boto3
import json
import re
from typing import List, Tuple, Dict
from models import Solution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DICT_KEY_LABELS = {
    'suspected_root_cause': '🔍 Suspected Root Cause',
    'root_cause': '🔍 Root Cause',
    'affected_components': '🏗️ Affected Components',
    'evidence_summary': '📋 Evidence Summary',
    'evidence': '📋 Evidence',
    'attack_pattern': '⚠️ Attack / Failure Pattern',
    'failure_pattern': '⚠️ Failure Pattern',
    'likely_attack_or_failure_pattern': '⚠️ Attack / Failure Pattern',
    'remediation_steps': '🔧 Remediation Steps',
    'remediation': '🔧 Remediation Steps',
    'prevention_recommendations': '🛡️ Prevention Recommendations',
    'prevention': '🛡️ Prevention Recommendations',
    'confidence': '📊 Confidence / Rationale',
    'rationale': '📊 Confidence / Rationale',
}


def _format_dict_solution(data: dict) -> str:
    """Convert nested dict AI response into readable text for the UI."""
    lines = []
    for key, value in data.items():
        label = _DICT_KEY_LABELS.get(key.lower(), key.replace('_', ' ').title())
        lines.append(f"**{label}**")
        if isinstance(value, list):
            for item in value:
                lines.append(f"  • {item}")
        else:
            lines.append(f"  {value}")
        lines.append("")
    return '\n'.join(lines)


class BedrockEnhancer:
    """Enhance solutions using AWS Bedrock"""
    
    def __init__(self, region: str = "us-east-1", model: str = "us.amazon.nova-micro-v1:0"):
        """
        Initialize Bedrock enhancer
        
        Args:
            region: AWS region
            model: Bedrock model ID
        """
        self.region = region
        self.model_id = model
        self.client = None
        
        try:
            self.client = boto3.client('bedrock-runtime', region_name=region)
        except Exception as e:
            print(f"Warning: Could not initialize Bedrock client: {e}")
    
    def is_available(self) -> bool:
        """Check if Bedrock is available"""
        return self.client is not None
    
    def enhance_solutions(
        self, 
        solutions: List[Solution], 
        log_examples: List[str] = None,
        ai_context = None,
        max_batch_size: int = 5
    ) -> Tuple[List[Solution], Dict]:
        """
        Enhance solutions using AWS Bedrock
        
        Args:
            solutions: List of basic solutions
            log_examples: Sample log entries for context (legacy, used if ai_context is None)
            ai_context: Structured AIContext from LogPreprocessor (preferred)
            max_batch_size: Maximum solutions per API call
            
        Returns:
            Tuple of (enhanced solutions, usage stats)
        """
        if not self.is_available():
            return solutions, {
                "ai_enhancement_used": False,
                "error": "Bedrock client not available"
            }
        
        enhanced_solutions = []
        total_tokens = 0
        total_cost = 0.0
        api_calls = 0
        
        # Process solutions in batches
        for i in range(0, len(solutions), max_batch_size):
            batch = solutions[i:i + max_batch_size]
            
            try:
                enhanced_batch, tokens, cost = self._enhance_batch(
                    batch, log_examples=log_examples, ai_context=ai_context
                )
                enhanced_solutions.extend(enhanced_batch)
                total_tokens += tokens
                total_cost += cost
                api_calls += 1
            except Exception as e:
                print(f"Error enhancing batch: {e}")
                # Truyền thẳng lỗi cho UI hiển thị thay vì âm thầm trả Basic Solutions
                return solutions, {
                    "ai_enhancement_used": False,
                    "error": f"Bedrock API Failed: {str(e)}"
                }
        
        # Safety check: verify that solutions were actually enhanced
        actually_enhanced = any(s.ai_enhanced for s in enhanced_solutions)
        
        if not actually_enhanced:
            return enhanced_solutions, {
                "ai_enhancement_used": False,
                "error": "Bedrock responded but AI could not parse the response. Solutions shown are basic (non-AI)."
            }
        
        usage_stats = {
            "ai_enhancement_used": True,
            "bedrock_model_used": self.model_id,
            "total_tokens_used": total_tokens,
            "estimated_total_cost": total_cost,
            "api_calls_made": api_calls
        }
        
        return enhanced_solutions, usage_stats
    
    def _enhance_batch(
        self, 
        solutions: List[Solution], 
        log_examples: List[str] = None,
        ai_context = None
    ) -> Tuple[List[Solution], int, float]:
        """Enhance a batch of solutions"""
        
        # Build prompt — prefer structured AIContext over flat examples
        prompt = self._build_prompt(solutions, log_examples=log_examples, ai_context=ai_context)
        
        # Call Bedrock API
        response = self._call_bedrock(prompt)
        
        # Parse response
        enhanced_solutions = self._parse_response(solutions, response)
        
        # Calculate tokens and cost
        tokens = response.get('usage', {}).get('total_tokens', 0)
        cost = self._calculate_cost(tokens)
        
        return enhanced_solutions, tokens, cost
    
    def _build_prompt(self, solutions: List[Solution], log_examples: List[str] = None, ai_context = None) -> str:
        """
        Build prompt for Bedrock.
        If ai_context (AIContext) is provided, builds a rich source-aware prompt.
        Otherwise falls back to legacy flat-examples prompt.
        """
        # ---- Rich prompt when AIContext is available ----
        if ai_context is not None:
            return self._build_rich_prompt(solutions, ai_context)
        
        # ---- Legacy fallback prompt ----
        prompt = "You are a log analysis expert. Enhance the following solutions with detailed, actionable recommendations.\n\n"
        
        if log_examples:
            prompt += "Sample log entries:\n"
            for i, example in enumerate(log_examples[:3], 1):
                prompt += f"{i}. {example}\n"
            prompt += "\n"
        
        prompt += "Solutions to enhance:\n\n"
        for i, solution in enumerate(solutions, 1):
            prompt += f"{i}. Problem: {solution.problem}\n"
            prompt += f"   Current solution: {solution.solution}\n"
            prompt += f"   Affected components: {', '.join(solution.affected_components)}\n\n"
        
        prompt += (
            "For each solution, provide:\n"
            "1. A detailed explanation of the root cause\n"
            "2. Step-by-step troubleshooting steps\n"
            "3. Specific commands or configurations to check\n"
            "4. Prevention strategies\n\n"
            "Format your response as JSON array with this structure:\n"
            "[\n"
            "  {\n"
            '    "problem": "original problem",\n'
            '    "enhanced_solution": "detailed solution text"\n'
            "  }\n"
            "]\n"
        )
        
        return prompt
    
    def _build_rich_prompt(self, solutions: List[Solution], ctx) -> str:
        """
        Build a source-aware prompt using structured AIContext.
        Produces a 7-part analysis output format for the demo.
        """
        # Source type label for the AI
        source_labels = {
            'vpc_flow': 'AWS VPC Flow Logs (network traffic records)',
            'cloudtrail': 'AWS CloudTrail (API audit logs)',
            'app': 'Application Logs (server/service logs)',
        }
        source_label = source_labels.get(ctx.source_type, 'Log data')
        
        prompt = (
            "You are an expert AWS security and log analysis engineer.\n"
            f"You are analyzing: {source_label}\n"
            f"Log group: {ctx.log_group_name}\n"
            f"Total logs retrieved: {ctx.total_logs_pulled} | "
            f"High-relevance logs: {ctx.total_logs_after_scoring}\n\n"
        )
        
        # Severity summary
        if ctx.severity_summary:
            prompt += "Severity distribution:\n"
            for sev, count in sorted(ctx.severity_summary.items(), key=lambda x: x[1], reverse=True):
                prompt += f"  {sev}: {count}\n"
            prompt += "\n"
        
        # Top error patterns
        if ctx.top_patterns:
            prompt += "Top error patterns (most frequent):\n"
            for i, p in enumerate(ctx.top_patterns, 1):
                prompt += f"  {i}. [{p['component']}] {p['pattern']} (count: {p['count']})\n"
            prompt += "\n"
        
        # Suspicious actors
        if ctx.suspicious_ips:
            prompt += "Suspicious IP addresses:\n"
            for item in ctx.suspicious_ips:
                prompt += f"  - {item['ip']} (seen {item['count']} times)\n"
            prompt += "\n"
        
        if ctx.suspicious_users:
            prompt += "Suspicious users/identities:\n"
            for item in ctx.suspicious_users:
                prompt += f"  - {item['user']} (seen {item['count']} times)\n"
            prompt += "\n"
        
        if ctx.suspicious_apis:
            prompt += "API actions observed:\n"
            for item in ctx.suspicious_apis:
                prompt += f"  - {item['api']} (count: {item['count']})\n"
            prompt += "\n"
        
        # Within-source hints
        if ctx.within_source_hints:
            prompt += "Correlation hints (within this log source):\n"
            for hint in ctx.within_source_hints:
                prompt += f"  - {hint}\n"
            prompt += "\n"
        
        # Representative samples
        if ctx.representative_samples:
            prompt += "Representative log samples (highest relevance):\n"
            for i, sample in enumerate(ctx.representative_samples, 1):
                prompt += f"  {i}. {sample}\n"
            prompt += "\n"
        
        # Detected issues to enhance
        prompt += "Detected issues to analyze:\n\n"
        for i, solution in enumerate(solutions, 1):
            prompt += f"{i}. Problem: {solution.problem}\n"
            prompt += f"   Current basic solution: {solution.solution}\n"
            prompt += f"   Affected components: {', '.join(solution.affected_components)}\n\n"
        
        # Output format instruction
        prompt += (
            "For each issue, provide a comprehensive analysis with these 7 parts:\n"
            "1. Suspected root cause\n"
            "2. Affected component(s)\n"
            "3. Evidence summary (reference specific log patterns, IPs, users, or APIs from above)\n"
            "4. Likely attack or failure pattern (e.g., brute-force, privilege escalation, resource exhaustion)\n"
            "5. Remediation steps (specific commands, AWS CLI, config changes)\n"
            "6. Prevention recommendations\n"
            "7. Brief confidence/rationale for your assessment\n\n"
            "IMPORTANT: Return ONLY a raw JSON array. Do not include markdown code blocks, conversational filler, or headers. Output starts with [ and ends with ].\n"
            "[\n"
            "  {\n"
            '    "problem": "original problem",\n'
            '    "enhanced_solution": "your full 7-part analysis as formatted text"\n'
            "  }\n"
            "]\n"
        )
        
        return prompt
    
    def _call_bedrock(self, prompt: str) -> dict:
        """Call Bedrock API"""
        # Prepare request body based on model
        if "claude" in self.model_id.lower():
            # Claude format
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "temperature": 0.3,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        else:
            # Nova format
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ],
                "inferenceConfig": {
                    "maxTokens": 4096,
                    "temperature": 0.3
                }
            }
        
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body)
        )
        
        response_body = json.loads(response['body'].read())
        
        # Extract text based on model format
        if "claude" in self.model_id.lower():
            text = response_body['content'][0]['text']
            usage = {
                'total_tokens': response_body['usage']['input_tokens'] + response_body['usage']['output_tokens']
            }
        else:
            # Nova format
            text = response_body['output']['message']['content'][0]['text']
            usage = {
                'total_tokens': response_body['usage']['inputTokens'] + response_body['usage']['outputTokens']
            }
        
        return {
            'text': text,
            'usage': usage
        }
    
    def _parse_response(self, original_solutions: List[Solution], response: dict) -> List[Solution]:
        """Parse Bedrock response and create enhanced solutions.
        Handles truncated JSON from max_tokens cutoff.
        """
        text = response['text']
        
        # Log raw response for debugging (first 500 chars)
        print(f"[Bedrock Response Preview] {text[:500]}")
        
        try:
            # 1. Look for markdown code blocks first
            json_text = ""
            code_block_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL)
            if code_block_match:
                json_text = code_block_match.group(1)
                print("[Bedrock Parse] Found JSON in markdown code block")
            else:
                # 2. Use regex to find the start of a JSON array: [{
                match = re.search(r'\[\s*\{', text, re.DOTALL)
                if match:
                    json_start = match.start()
                    json_end = text.rfind(']') + 1
                    if json_end > json_start:
                        json_text = text[json_start:json_end]
                else:
                    # 3. Fallback to simple find
                    json_start = text.find('[')
                    json_end = text.rfind(']') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_text = text[json_start:json_end]

            if json_text:
                enhanced_data = self._safe_json_loads(json_text)
                
                if enhanced_data is not None:
                    enhanced_solutions = []
                    for i, solution in enumerate(original_solutions):
                        if i < len(enhanced_data):
                            raw_val = enhanced_data[i].get('enhanced_solution', solution.solution)
                            # AI sometimes returns a nested dict instead of a string
                            if isinstance(raw_val, dict):
                                enhanced_text = _format_dict_solution(raw_val)
                            elif isinstance(raw_val, list):
                                enhanced_text = '\n'.join(str(v) for v in raw_val)
                            else:
                                enhanced_text = str(raw_val)
                        else:
                            enhanced_text = solution.solution
                        
                        enhanced_solution = Solution(
                            problem=solution.problem,
                            solution=enhanced_text,
                            issue_type=solution.issue_type,
                            affected_components=solution.affected_components,
                            ai_enhanced=True,
                            tokens_used=response.get('usage', {}).get('total_tokens', 0) // len(original_solutions),
                            estimated_cost=self._calculate_cost(response.get('usage', {}).get('total_tokens', 0)) / len(original_solutions)
                        )
                        enhanced_solutions.append(enhanced_solution)
                    
                    return enhanced_solutions
                else:
                    print(f"[Bedrock Parse Warning] Could not parse JSON even after repair. Text: {json_text[:500]}")
                    return original_solutions
            else:
                # No JSON array found at all
                print(f"[Bedrock Parse Warning] No JSON array found in response. Full text: {text[:1000]}")
                return original_solutions
        
        except Exception as e:
            print(f"[Bedrock Parse Error] Unexpected error: {e}")
            return original_solutions
    
    def _safe_json_loads(self, text: str):
        """
        Try to parse JSON. If it fails (likely truncated output),
        attempt to repair by closing open brackets/braces.
        """
        # Attempt 1: direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Attempt 2: response might be truncated — try closing open structures
        repaired = text.rstrip()
        # Remove any trailing incomplete string value
        # Find last complete object by looking for last '}'
        last_brace = repaired.rfind('}')
        if last_brace > 0:
            repaired = repaired[:last_brace + 1]
            # Close the array
            if not repaired.rstrip().endswith(']'):
                repaired = repaired.rstrip().rstrip(',') + ']'
            try:
                result = json.loads(repaired)
                print(f"[Bedrock Parse] Successfully repaired truncated JSON ({len(result)} items)")
                return result
            except json.JSONDecodeError:
                pass
        
        print(f"[Bedrock Parse] JSON repair failed. Raw: {text[:300]}")
        return None
    
    def _calculate_cost(self, tokens: int) -> float:
        """Calculate estimated cost based on tokens"""
        # Nova Micro pricing: $0.035 per 1M input tokens, $0.14 per 1M output tokens
        # Assume 50/50 split for simplicity
        if "nova-micro" in self.model_id.lower():
            input_cost_per_1m = 0.035
            output_cost_per_1m = 0.14
            avg_cost_per_1m = (input_cost_per_1m + output_cost_per_1m) / 2
        # Claude Haiku pricing: $0.25 per 1M input tokens, $1.25 per 1M output tokens
        elif "haiku" in self.model_id.lower():
            input_cost_per_1m = 0.25
            output_cost_per_1m = 1.25
            avg_cost_per_1m = (input_cost_per_1m + output_cost_per_1m) / 2
        # Claude Sonnet pricing: $3 per 1M input tokens, $15 per 1M output tokens
        elif "sonnet" in self.model_id.lower():
            input_cost_per_1m = 3.0
            output_cost_per_1m = 15.0
            avg_cost_per_1m = (input_cost_per_1m + output_cost_per_1m) / 2
        else:
            # Default to Nova Micro pricing
            avg_cost_per_1m = 0.0875
        
        return (tokens / 1_000_000) * avg_cost_per_1m
