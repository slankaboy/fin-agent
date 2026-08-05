
import requests
import pandas as pd
import json
import os
from datetime import datetime

URL = "https://api.jiucaishuo.com/v2/guzhi-new2/peg-sg2"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": "https://www.jiucaishuo.com/",
    "Origin": "https://www.jiucaishuo.com",
}

SIGN_PARAMS = {
    "tirgkjfs": "d2",
    "abiokytke": "4f",
    "u54rg5d": "cf",
    "kf54ge7": "b",
    "tiklsktr4": "2",
    "lksytkjh": "c894",
    "sbnoywr": "c2",
    "bgd7h8tyu54": "5c",
    "y654b5fs3tr": "8",
    "bioduytlw": "3",
    "bd4uy742": "e",
    "h67456y": "bc8",
    "bvytikwqjk": "5c",
    "ngd4uy551": "c8",
    "bgiuytkw": "b6",
    "nd354uy4752": "9",
    "ghtoiutkmlg": "896",
    "bd24y6421f": "2c",
    "tbvdiuytk": "b",
    "ibvytiqjek": "05",
    "jnhf8u5231": "b6",
    "fjlkatj": "cf2",
    "hy5641d321t": "ce",
    "iogojti": "c",
    "ngd4yut78": "96",
    "nkjhrew": "e",
    "yt447e13f": "7",
    "n3bf4uj7y7": "8",
    "nbf4uj7y432": "4f",
    "yi854tew": "29",
    "h13ey474": "29b",
    "quikgdky": "d1"
}


def fetch_peg_data(output_dir="mock_files"):
    """获取 PEG 模型数据"""
    os.makedirs(output_dir, exist_ok=True)
    current_timestamp = int(datetime.now().timestamp() * 1000)

    # 使用用户提供的原始 payload
    payload = {
        "type": "h5",
        "version": "2.5.9",
        "ss": "",
        "act_time": 1785912462606,  # 使用用户提供的固定时间戳测试
        "tirgkjfs": "d2",
        "abiokytke": "4f",
        "u54rg5d": "cf",
        "kf54ge7": "b",
        "tiklsktr4": "2",
        "lksytkjh": "c894",
        "sbnoywr": "c2",
        "bgd7h8tyu54": "5c",
        "y654b5fs3tr": "8",
        "bioduytlw": "3",
        "bd4uy742": "e",
        "h67456y": "bc8",
        "bvytikwqjk": "5c",
        "ngd4uy551": "c8",
        "bgiuytkw": "b6",
        "nd354uy4752": "9",
        "ghtoiutkmlg": "896",
        "bd24y6421f": "2c",
        "tbvdiuytk": "b",
        "ibvytiqjek": "05",
        "jnhf8u5231": "b6",
        "fjlkatj": "cf2",
        "hy5641d321t": "ce",
        "iogojti": "c",
        "ngd4yut78": "96",
        "nkjhrew": "e",
        "yt447e13f": "7",
        "n3bf4uj7y7": "8",
        "nbf4uj7y432": "4f",
        "yi854tew": "29",
        "h13ey474": "29b",
        "quikgdky": "d1"
    }

    try:
        print("正在请求 PEG 模型数据...")
        response = requests.post(URL, data=payload, headers=HEADERS, timeout=15)
        response.raise_for_status()
        result = response.json()

        print(f"响应状态: {result.get('code', 'N/A')}")
        print(f"响应消息: {result.get('message', 'N/A')}")

        data = result.get('data', {})
        if not data:
            print("⚠️ data 字段为空")
            return None

        # PEG 接口返回的是图表数据结构
        title = data.get('title', '')
        desc = data.get('desc', '')
        x_label = data.get('_x', '')
        y_label = data.get('_y', '')
        tuli_x = data.get('tuli_x', '')
        tuli_y = data.get('tuli_y', '')
        update_time = data.get('update_time', '')
        series = data.get('series', [])

        if not series:
            print("⚠️ series 字段为空")
            return None

        print(f"✓ 标题: {title}")
        print(f"✓ 更新时间: {update_time}")
        print(f"✓ X轴: {x_label} ({tuli_x})")
        print(f"✓ Y轴: {y_label} ({tuli_y})")
        print(f"✓ 共 {len(series)} 条记录")

        # 扁平化数据
        flat_records = []
        for item in series:
            x1_val = item.get('x1', '')
            y1_val = item.get('y1', '')
            flat_item = {
                'name': item.get('name', ''),
                '2y_profit_growth_rate': f"{x1_val}%" if x1_val != '' else '',
                'pred_2y_pe': f"{y1_val}" if y1_val != '' else '',
                'update_time': update_time,
                'title': title,
            }
            flat_records.append(flat_item)

        df = pd.DataFrame(flat_records)
        df['_fetch_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 添加描述信息到单独的元数据文件
        meta_info = {
            'title': title,
            'desc': desc.replace('<br>', ' '),
            'x_label': x_label,
            'y_label': y_label,
            'tuli_x': tuli_x,
            'tuli_y': tuli_y,
            'update_time': update_time,
        }

        date_str = datetime.now().strftime('%Y%m%d')
        csv_filename = os.path.join(output_dir, f"PEG模型_{date_str}.csv")
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"✓ 数据已保存到: {csv_filename}")

        # 保存元数据
        meta_filename = os.path.join(output_dir, f"PEG模型_元数据_{date_str}.json")
        with open(meta_filename, 'w', encoding='utf-8') as f:
            json.dump(meta_info, f, ensure_ascii=False, indent=2)
        print(f"✓ 元数据已保存到: {meta_filename}")

        # 显示数据预览
        print("\n=== PEG 模型数据预览 ===")
        print(df.head(10).to_string(index=False))

        return df

    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析失败: {e}")
        return None
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        return None


if __name__ == "__main__":
    print("开始获取 PEG 模型数据...")
    print("=" * 50)

    df = fetch_peg_data(output_dir="mock_files")

    if df is not None:
        print(f"\n总计: {len(df)} 条记录")
    print("\n处理完成!")
