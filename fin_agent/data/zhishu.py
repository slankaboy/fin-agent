
titles = [
            {
                "id": 6,
                "name": "自选",
                "types": 2
            },
            {
                "id": 8,
                "name": "精选",
                "types": 2
            },
            {
                "id": 2,
                "name": "规模",
                "types": 2
            },
            {
                "id": 1,
                "name": "行业",
                "types": 2
            },
            {
                "id": 9,
                "name": "策略",
                "types": 2
            },
            {
                "id": 3,
                "name": "主题",
                "types": 2
            },
            {
                "id": 5,
                "name": "境外",
                "types": 2
            },
            {
                "id": 7,
                "name": "其他",
                "types": 1
            }]



import requests
import pandas as pd
import json
import os
from datetime import datetime
import time

# 固定的请求参数（签名相关字段）
# 指数估值查询签名参数（types=2）
SIGN_PARAMS_VALUATION = {
    "tirgkjfs": "d8",
    "abiokytke": "3c",
    "u54rg5d": "fc",
    "kf54ge7": "8",
    "tiklsktr4": "8",
    "lksytkjh": "5d24",
    "sbnoywr": "e7",
    "bgd7h8tyu54": "77",
    "y654b5fs3tr": "f",
    "bioduytlw": "0",
    "bd4uy742": "d",
    "h67456y": "f5d",
    "bvytikwqjk": "77",
    "ngd4uy551": "5d",
    "bgiuytkw": "1c",
    "nd354uy4752": "d",
    "ghtoiutkmlg": "f31",
    "bd24y6421f": "78",
    "tbvdiuytk": "f",
    "ibvytiqjek": "ce",
    "jnhf8u5231": "1c",
    "fjlkatj": "fc1",
    "hy5641d321t": "8d",
    "iogojti": "8",
    "ngd4yut78": "31",
    "nkjhrew": "d",
    "yt447e13f": "c",
    "n3bf4uj7y7": "d",
    "nbf4uj7y432": "3c",
    "yi854tew": "6d",
    "h13ey474": "6d8",
    "quikgdky": "93"
}

# 行业涨跌查询签名参数（types=1, category=-1）
SIGN_PARAMS_INDUSTRY = {
    "tirgkjfs": "c0",
    "abiokytke": "db",
    "u54rg5d": "84",
    "kf54ge7": "7",
    "tiklsktr4": "0",
    "lksytkjh": "a8ad",
    "sbnoywr": "e6",
    "bgd7h8tyu54": "cd",
    "y654b5fs3tr": "1",
    "bioduytlw": "2",
    "bd4uy742": "7",
    "h67456y": "4a8",
    "bvytikwqjk": "cd",
    "ngd4uy551": "a8",
    "bgiuytkw": "04",
    "nd354uy4752": "a",
    "ghtoiutkmlg": "17d",
    "bd24y6421f": "6f",
    "tbvdiuytk": "4",
    "ibvytiqjek": "6d",
    "jnhf8u5231": "04",
    "fjlkatj": "849",
    "hy5641d321t": "f7",
    "iogojti": "f",
    "ngd4yut78": "7d",
    "nkjhrew": "7",
    "yt447e13f": "e",
    "n3bf4uj7y7": "8",
    "nbf4uj7y432": "db",
    "yi854tew": "ba",
    "h13ey474": "ba7",
    "quikgdky": "f9"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": "https://www.jiucaishuo.com/",
    "Origin": "https://www.jiucaishuo.com",
}

URL = "https://apiv2.jiucaishuo.com/indexvaluation/industry/group-list"


def fetch_page(category, types, page=1, sign_params=None):
    """请求单页数据，返回 (records, head, update_time) 或 (None, None, None)"""
    if sign_params is None:
        sign_params = SIGN_PARAMS_VALUATION

    current_timestamp = int(datetime.now().timestamp())
    payload = {
        "category": category,
        "types": str(types),
        "p": page,
        "field": "",
        "order_by": "",
        "type": "h5",
        "version": "2.5.9",
        "ss": "",
        "act_time": current_timestamp,
    }
    payload.update(sign_params)

    try:
        response = requests.post(URL, data=payload, headers=HEADERS, timeout=15)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException as e:
        print(f"  ❌ 请求失败: {e}")
        return None, None, None
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON解析失败: {e}")
        return None, None, None

    data = result.get('data', {})
    if isinstance(data, dict):
        records = data.get('list', [])
        head = data.get('head', [])
        update_time = data.get('update_time', '')
    elif isinstance(data, list):
        records = data
        head = []
        update_time = ''
    else:
        records = []
        head = []
        update_time = ''

    return records, head, update_time


def parse_head(head):
    """解析 head，按 id 排序，忽略第一个列，用 val 做表头名称"""
    if not head:
        return []
    sorted_head = sorted(head, key=lambda h: h.get('id', 0))
    # 忽略第一个列，使用 val 作为表头名
    return [h.get('val', '') for h in sorted_head[1:]]


def flatten_records_with_head(records, head_names, update_time, category_name=''):
    """使用 head 表头名称将嵌套数据扁平化"""
    flat_records = []
    for item in records:
        flat_item = {
            'code': item.get('gu_code', ''),
            'name': item.get('gu_name', ''),
            'level': item.get('level', ''),
            'tag': item.get('tag', ''),
        }
        if category_name:
            flat_item['category'] = category_name
        flat_item['update_time'] = update_time

        # 跳过第一个元素（对应 head 第一列），从第二个开始
        item_list = item.get('list', [])[1:]
        for i, col_name in enumerate(head_names):
            if i < len(item_list):
                flat_item[col_name] = item_list[i].get('val', '')
            else:
                flat_item[col_name] = ''
        flat_records.append(flat_item)
    return flat_records


def fetch_industry_data(max_pages=10):
    """获取行业指数涨跌数据 (category=-1, types=1)"""
    print("\n" + "=" * 60)
    print("正在获取行业指数涨跌数据...")
    print("=" * 60)

    all_flat = []
    head_names = []
    update_time = ''
    for page in range(1, max_pages + 1):
        print(f"  获取第 {page} 页...")
        records, head, ut = fetch_page(-1, 1, page=page, sign_params=SIGN_PARAMS_INDUSTRY)
        if not records:
            print(f"  第 {page} 页无数据，停止翻页")
            break
        if not update_time and ut:
            update_time = ut
        if not head_names and head:
            head_names = parse_head(head)
            print(f"  表头: {head_names}")

        flat = flatten_records_with_head(records, head_names, update_time)
        all_flat.extend(flat)

        print(f"  第 {page} 页: {len(records)} 条记录")
        time.sleep(0.5)

    return all_flat, head_names, update_time


def fetch_category_data(category_id, types, category_name, max_pages=10, sign_params=None):
    """获取某个分类的所有页数据"""
    if sign_params is None:
        sign_params = SIGN_PARAMS_VALUATION

    all_flat = []
    head_names = []
    update_time = ''
    for page in range(1, max_pages + 1):
        print(f"  获取第 {page} 页...")
        records, head, ut = fetch_page(category_id, types, page=page, sign_params=sign_params)
        if not records:
            print(f"  第 {page} 页无数据，停止翻页")
            break
        if not update_time and ut:
            update_time = ut
        if not head_names and head:
            head_names = parse_head(head)
            print(f"  表头: {head_names}")

        flat = flatten_records_with_head(records, head_names, update_time, category_name)
        all_flat.extend(flat)
        print(f"  第 {page} 页: {len(records)} 条记录")
        time.sleep(0.5)
    return all_flat, head_names, update_time


def fetch_all_categories(output_dir="mock_files"):
    """迭代 titles 获取所有分类数据，分别保存为 CSV"""
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d')
    summary = []

    for title in titles:
        category_id = title['id']
        types = title['types']
        name = title['name']
        print(f"\n{'='*50}")
        print(f"正在获取分类: {name} (id={category_id}, types={types})")
        print(f"{'='*50}")

        flat_records, head_names, update_time = fetch_category_data(category_id, types, name, max_pages=1)
        if not flat_records:
            print(f"⚠️ 分类 [{name}] 无数据，跳过")
            summary.append({'category': name, 'count': 0, 'file': ''})
            continue

        df = pd.DataFrame(flat_records)
        df['_fetch_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        csv_filename = os.path.join(output_dir, f"指数估值_{name}_{date_str}.csv")
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"✓ [{name}] 共 {len(df)} 条记录，已保存到: {csv_filename}")

        summary.append({'category': name, 'count': len(df), 'file': csv_filename})

    return summary


def fetch_and_save_industry(output_dir="mock_files"):
    """获取行业指数涨跌数据并保存"""
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d')

    flat_records, head_names, update_time = fetch_industry_data(max_pages=10)
    if not flat_records:
        print("⚠️ 行业涨跌数据为空")
        return {'category': '行业涨跌', 'count': 0, 'file': ''}

    df = pd.DataFrame(flat_records)
    df['_fetch_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    csv_filename = os.path.join(output_dir, f"行业指数涨跌_{date_str}.csv")
    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"✓ [行业涨跌] 共 {len(df)} 条记录，已保存到: {csv_filename}")

    print("\n=== 行业涨跌数据预览 ===")
    print(df.head(10).to_string(index=False))

    return {'category': '行业涨跌', 'count': len(df), 'file': csv_filename}


if __name__ == "__main__":
    print("开始获取韭财说数据...")
    print("=" * 60)
    
    # 1. 获取行业指数涨跌数据 (使用新参数)
    print("\n【1/2】获取行业指数涨跌数据...")
    industry_result = fetch_and_save_industry(output_dir="mock_files")
    
    # 2. 获取所有分类指数估值数据
    print("\n【2/2】获取所有分类指数估值数据...")
    valuation_summary = fetch_all_categories(output_dir="mock_files")
    
    # 汇总输出
    print("\n" + "=" * 60)
    print("获取完成！汇总如下：")
    print("=" * 60)
    
    print("\n📈 行业指数涨跌:")
    print(f"  记录数: {industry_result['count']}")
    print(f"  文件: {industry_result['file']}")
    
    print("\n📊 指数估值分类:")
    total = industry_result['count']
    for s in valuation_summary:
        print(f"  {s['category']:<8} | {s['count']:>4} 条 | {s['file']}")
        total += s['count']
    
    print(f"\n总计: {total} 条记录")
    print("\n处理完成!")
