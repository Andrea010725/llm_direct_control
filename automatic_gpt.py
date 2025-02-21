from dataclasses import dataclass
from typing import Optional, Tuple, List
import subprocess
import time
from pathlib import Path

@dataclass
class ScriptOutput:
    path_decision: str
    speed_decision: str
    explanation: str

class ScriptRunner:
    def __init__(self, script_path: str, conda_env: str):
        self.script_path = Path(script_path)
        self.conda_env = conda_env
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """验证输入参数的有效性"""
        if not self.script_path.exists():
            raise FileNotFoundError(f"脚本文件不存在：{self.script_path}")

    def _parse_output(self, output: str) -> ScriptOutput:
        """解析脚本输出的XML格式内容"""
        try:
            vector_content = output.split('<VECTOR>')[1].split('</VECTOR>')[0].strip()
            explanation = output.split('<EXPLANATION>')[1].split('</EXPLANATION>')[0].strip()
            
            path_decision = vector_content.split('<PATHVECTOR>')[1].split('</PATHVECTOR>')[0].strip()
            speed_decision = vector_content.split('<SPEEDVECTOR>')[1].split('</SPEEDVECTOR>')[0].strip()
            
            return ScriptOutput(
                path_decision=path_decision,
                speed_decision=speed_decision,
                explanation=explanation
            )
        except IndexError as e:
            raise ValueError("XML输出格式解析失败") from e

    def execute(self) -> Tuple[Optional[ScriptOutput], float]:
        """执行脚本并返回解析后的结果和执行时间"""
        try:
            cmd = f'cmd /c "conda activate {self.conda_env} && python "{self.script_path}""'
            start_time = time.time()
            
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True
            )
            
            execution_time = time.time() - start_time

            if result.returncode != 0:
                print(f"脚本执行失败，错误信息：{result.stderr}")
                return None, execution_time

            output = self._parse_output(result.stdout.strip())
            print(f"脚本执行成功，用时：{execution_time:.2f}秒")
            print(f"决策结果：路径={output.path_decision}, "
                  f"速度={output.speed_decision}, "
                  f"解释={output.explanation}")
            
            return output, execution_time

        except Exception as e:
            print(f"执行过程中发生错误：{str(e)}")
            return None, 0

def monitor_script(script_path: str, conda_env: str) -> Optional[ScriptOutput]:
    """监控并执行脚本"""
    runner = ScriptRunner(script_path, conda_env)
    output, _ = runner.execute()
    return output

def get_gpt_result():
    TARGET_SCRIPT = "D:\\Github仓库\\llm_direct_control\\openai_interaction.py"
    CONDA_ENV = "chatgpt"
    
    # 执行脚本
    result = monitor_script(TARGET_SCRIPT, CONDA_ENV)
    return result

if __name__ == "__main__":
    # 配置参数
    TARGET_SCRIPT = "D:\\Github仓库\\llm_direct_control\\openai_interaction.py"
    CONDA_ENV = "chatgpt"
    
    # 执行脚本
    result = monitor_script(TARGET_SCRIPT, CONDA_ENV)
    print(f"最终决策结果：{result.path_decision}, {result.speed_decision}, {result.explanation}")
