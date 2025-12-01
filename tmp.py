import gymnasium as gym
import time

try:
    # 렌더링 모드를 'human'으로 지정하여 환경 생성
    env = gym.make("Walker2d-v5", render_mode="human")
    
    print("환경 생성 성공. 5초간 렌더링 테스트를 진행합니다.")
    
    # 환경 초기화
    env.reset()
    
    # 100 스텝 동안 렌더링
    for _ in range(100):
        action = env.action_space.sample() # 무작위 행동 샘플링
        _, _, terminated, truncated, _ = env.step(action)
        
        # 렌더링이 자동으로 될 때까지 잠시 대기 (필요 시)
        # env.render() # <-- render_mode="human" 사용 시 일반적으로 필요 없음
        
        if terminated or truncated:
            env.reset()
            
        time.sleep(0.01) # 너무 빠르게 지나가는 것을 방지

    print("테스트 완료.")

except Exception as e:
    print(f"렌더링 테스트 실패: {e}")
    
finally:
    if 'env' in locals():
        env.close()