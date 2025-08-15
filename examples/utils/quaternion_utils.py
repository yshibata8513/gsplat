"""
Quaternion utility functions for dynamic Gaussian Splatting

クォータニオンの基本演算と座標変換を提供。
主に動的物体の回転処理に使用。
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    クォータニオンの積を計算
    
    Args:
        q1, q2: クォータニオン [4,] (w, x, y, z) format
        
    Returns:
        result: q1 * q2 [4,]
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return torch.stack([w, x, y, z])


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    クォータニオンから回転行列に変換
    
    Args:
        q: クォータニオン [4,] (w, x, y, z) format, 正規化済み
        
    Returns:
        R: 回転行列 [3, 3]
    """
    # クォータニオンを正規化
    q = F.normalize(q, dim=-1)
    
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # 回転行列の成分を計算
    R = torch.zeros(3, 3, device=q.device, dtype=q.dtype)
    
    R[0, 0] = 1 - 2*(y*y + z*z)
    R[0, 1] = 2*(x*y - w*z)
    R[0, 2] = 2*(x*z + w*y)
    
    R[1, 0] = 2*(x*y + w*z)
    R[1, 1] = 1 - 2*(x*x + z*z)
    R[1, 2] = 2*(y*z - w*x)
    
    R[2, 0] = 2*(x*z - w*y)
    R[2, 1] = 2*(y*z + w*x)
    R[2, 2] = 1 - 2*(x*x + y*y)
    
    return R


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    回転行列からクォータニオンに変換
    
    Args:
        R: 回転行列 [3, 3]
        
    Returns:
        q: クォータニオン [4,] (w, x, y, z) format, 正規化済み
    """
    # Shepperdの方法を使用（数値的に安定）
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2  # s = 4 * w
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * x
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * y
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * z
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    q = torch.stack([w, x, y, z])
    return F.normalize(q, dim=-1)


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """
    クォータニオンの共役を計算
    
    Args:
        q: クォータニオン [4,] (w, x, y, z)
        
    Returns:
        q_conj: 共役クォータニオン [4,] (w, -x, -y, -z)
    """
    return torch.stack([q[0], -q[1], -q[2], -q[3]])


def quaternion_inverse(q: torch.Tensor) -> torch.Tensor:
    """
    クォータニオンの逆元を計算（正規化済みクォータニオンの場合は共役と同じ）
    
    Args:
        q: 正規化済みクォータニオン [4,]
        
    Returns:
        q_inv: 逆クォータニオン [4,]
    """
    return quaternion_conjugate(q)


def rotate_point_by_quaternion(point: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    点をクォータニオンで回転
    
    Args:
        point: 3D点 [3,]
        q: 回転クォータニオン [4,] (w, x, y, z), 正規化済み
        
    Returns:
        rotated_point: 回転された点 [3,]
    """
    # 点を純クォータニオンに変換 (0, x, y, z)
    point_quat = torch.cat([torch.zeros(1, device=point.device), point])
    
    # 回転: q * point_quat * q^(-1)
    q_inv = quaternion_conjugate(q)
    temp = quaternion_multiply(q, point_quat)
    result_quat = quaternion_multiply(temp, q_inv)
    
    return result_quat[1:]  # 実部を除いて座標部分を返す


def quaternion_to_euler(q: torch.Tensor) -> torch.Tensor:
    """
    クォータニオンからオイラー角（roll, pitch, yaw）に変換
    
    Args:
        q: クォータニオン [4,] (w, x, y, z)
        
    Returns:
        euler: オイラー角 [3,] (roll, pitch, yaw) in radians
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w*x + y*z)
    cosr_cosp = 1 - 2 * (x*x + y*y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w*y - z*x)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.copysign(torch.pi / 2, sinp),  # Use 90 degrees if out of range
        torch.asin(sinp)
    )
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w*z + x*y)
    cosy_cosp = 1 - 2 * (y*y + z*z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    
    return torch.stack([roll, pitch, yaw])


def euler_to_quaternion(euler: torch.Tensor) -> torch.Tensor:
    """
    オイラー角からクォータニオンに変換
    
    Args:
        euler: オイラー角 [3,] (roll, pitch, yaw) in radians
        
    Returns:
        q: クォータニオン [4,] (w, x, y, z)
    """
    roll, pitch, yaw = euler[0], euler[1], euler[2]
    
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    q = torch.stack([w, x, y, z])
    return F.normalize(q, dim=-1)


def quaternion_slerp(q1: torch.Tensor, q2: torch.Tensor, t: float) -> torch.Tensor:
    """
    2つのクォータニオン間の球面線形補間（SLERP）
    
    Args:
        q1, q2: 補間するクォータニオン [4,]
        t: 補間パラメータ [0, 1]
        
    Returns:
        q_interp: 補間されたクォータニオン [4,]
    """
    # 正規化
    q1 = F.normalize(q1, dim=-1)
    q2 = F.normalize(q2, dim=-1)
    
    # 内積
    dot = torch.sum(q1 * q2)
    
    # 最短パスを選ぶ
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # ほぼ同じ方向の場合は線形補間
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return F.normalize(result, dim=-1)
    
    # SLERP
    theta_0 = torch.acos(torch.clamp(torch.abs(dot), 0, 1))
    sin_theta_0 = torch.sin(theta_0)
    
    theta = theta_0 * t
    sin_theta = torch.sin(theta)
    
    s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return s0 * q1 + s1 * q2


# テスト用関数
def test_quaternion_utils():
    """クォータニオンユーティリティ関数のテスト"""
    import math
    
    print("Testing quaternion utilities...")
    
    # 単位クォータニオン（回転なし）
    q_identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
    
    # Z軸周り90度回転のクォータニオン
    q_90z = torch.tensor([math.cos(math.pi/4), 0.0, 0.0, math.sin(math.pi/4)])
    
    # 回転行列に変換
    R = quaternion_to_rotation_matrix(q_90z)
    print(f"90° Z-rotation matrix:\n{R}")
    
    # 逆変換
    q_recovered = rotation_matrix_to_quaternion(R)
    print(f"Original quaternion: {q_90z}")
    print(f"Recovered quaternion: {q_recovered}")
    
    # 点の回転テスト
    point = torch.tensor([1.0, 0.0, 0.0])
    rotated = rotate_point_by_quaternion(point, q_90z)
    print(f"Point [1,0,0] rotated 90° around Z: {rotated}")
    
    # SLERP テスト
    q_interp = quaternion_slerp(q_identity, q_90z, 0.5)
    print(f"SLERP halfway: {q_interp}")
    
    print("Quaternion utilities test completed.")


if __name__ == "__main__":
    test_quaternion_utils()