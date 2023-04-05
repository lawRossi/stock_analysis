import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
import os
from utils import load_reports_list, load_financial_reports


plot.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plot.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def compute_increment(numbers, return_number=False):
    increments = [(numbers[i] - numbers[i-1]) * 100 / numbers[i-1] if numbers[i-1] > 0 else None for i in range(1, len(numbers))]
    if not return_number:
        increments = ["%d%%" % increment if increment is not None else "" for increment in increments]
    return increments


def plot_line_chart(data, items, title, ylabel="金额(亿)", xlabel="年份", with_increment=False, save_path=None):
    plot.figure(figsize=(6, 4), dpi=100)
    for item in items:
        sub_data = data.loc[item]
        sub_data = sub_data.fillna(0)
        plot.plot(sub_data, label=item)
        if with_increment:
            increments = compute_increment(sub_data)
            indexes = list(sub_data.index)
            for i in range(1, len(indexes)):
                plot.text(indexes[i], sub_data[i], increments[i-1])
    plot.ylabel(ylabel)
    plot.xlabel(xlabel)
    plot.legend(fontsize="x-small")
    plot.title(title)
    if save_path is not None:
        plot.savefig(save_path, format="png")
        plot.cla()
    else:
        plot.show()


def plot_double_y_chart(data, item, title, ylabel="金额(亿)", xlabel="年份", save_path=None):
    _, ax = plot.subplots(1, 1, figsize=(6, 4), dpi=100)
    ax_ = ax.twinx()
    sub_data = data.loc[item].fillna(0)
    increments = compute_increment(sub_data, True)
    increments.insert(0, None)
    indexes = sub_data.index.astype(int)
    ax.bar(indexes, sub_data.values, label=item)
    for idx, data in zip(indexes, sub_data.values):
        ax.text(idx-0.4, data, "%.2f" % data, fontsize="small")
    ax_.plot(indexes, increments, label="增长率", color="orange")
    plot.xticks(indexes)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax_.set_ylabel("增长率")
    plot.title(title)
    if save_path is not None:
        plot.savefig(save_path, format="png")
        plot.cla()
    else:
        plot.show()


def plot_pie_chart(data, labels, title, save_path=None):
    s = sum(data)
    sizes = [item/s for item in data]
    sizes, labels = sort_data(sizes, labels)
    max_idx = np.argmax(sizes)
    explode = [0] * len(labels)
    explode[max_idx] = 0.1
    plot.figure(figsize=(6, 6), dpi=100)
    _, label_texts, _ = plot.pie(sizes, labels=labels, explode=explode, autopct="%0.1f%%", labeldistance=1.05, radius=0.8)
    for label_text in label_texts:
        label_text.set_size(9)
    plot.title(title)
    if save_path is not None:
        plot.savefig(save_path, format="png")
        plot.cla()
        plot.close()
    else:
        plot.show()


def sort_data(sizes, labels):
    """
    按大小间隔排序，以免标签重合
    """
    data = sorted(zip(sizes, labels), key=lambda x: x[0])
    i = 0
    j = (len(data) + 1) // 2
    sorted_data = []
    while j < len(data):
        sorted_data.append(data[i])
        sorted_data.append(data[j])
        i += 1
        j += 1
    if len(data) % 2 != 0:
        sorted_data.append(data[i])
    sizes, labels = zip(*sorted_data)
    return sizes, labels


def plot_stack_bar_chart(data, items, labels, title, ylabel="金额(亿)", xlabel="年份", save_path=None):
    columns = data.columns
    bottom = np.zeros(len(columns))
    for item, label in zip(items, labels):
        sub_data = data.loc[item]
        sub_data.fillna(0)
        plot.bar(columns, sub_data, bottom=bottom, label=label)
        bottom += sub_data

    plot.ylim(0, max(bottom) * 1.2)
    plot.legend(loc="upper left", fontsize="small")
    plot.xlabel(xlabel)
    plot.ylabel(ylabel)
    plot.title(title)
    if save_path is not None:
        plot.savefig(save_path, format="png")
        plot.cla()
    else:
        plot.show()


def plot_adjacent_bar_chart(data, items, labels, title, ylabel="金额(亿)", xlabel="年份", save_path=None):
    columns = data.columns.astype(int)
    idx = 0
    total_width = 0.8
    width = total_width / len(items)
    x = columns - (total_width - width) / 2
    max_value = 0
    min_value = 0
    for item, label in zip(items, labels):
        sub_data = data.loc[item]
        max_value = max(max_value, sub_data.max())
        min_value = min(min_value, sub_data.min())
        sub_data.fillna(0)
        plot.bar(x+idx*width, sub_data, width=width, label=label)
        for i, j in zip(x, sub_data):
            offset = 0.2 if j < 1000 else 0.25
            fs = "small" if j < 1000 else "x-small"
            plot.text(i+idx*width-offset, j, "%0.1f" % j, fontsize=fs)
        idx += 1
    plot.ylim(min_value*1.1, max_value*1.2)
    plot.legend(loc="upper left", fontsize="x-small")
    plot.xlabel(xlabel)
    plot.ylabel(ylabel)
    plot.title(title)
    if save_path is not None:
        plot.savefig(save_path, format="png")
        plot.cla()
    else:
        plot.show()


def plot_overlap_bar_chart(data, items, labels, title, ylabel="金额(亿)", xlabel="年份", save_path=None, with_increment=True):
    _, ax = plot.subplots(1, 1, figsize=(6, 4), dpi=100)
    if with_increment:
        ax_ = ax.twinx()
    colors = ["green", "purple", "red"]
    for i, (item, label) in enumerate(zip(items, labels)):
        ax.bar(data.columns, data.loc[item], label=label)
        for column, value in zip(data.columns, data.loc[item]):
            ax.text(column, value, str(value), horizontalalignment="center")
        if with_increment:
            increments = compute_increment(data.loc[item], True)
            increments.insert(0, None)
            ax_.plot(data.columns, increments, label=f"{label}增长率", color=colors[i])

    ax.legend(loc="upper left", fontsize="x-small")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if with_increment:
        ax_.legend(loc="upper center", fontsize="x-small")
        ax_.set_ylabel("增长率(%)")

    plot.title(title)
    if save_path is not None:
        plot.savefig(save_path, format="png")
        plot.cla()
    else:
        plot.show()


def plot_reports(report_file_path, save_dir=None):
    main_stats_report, profit_report, capital_debt_report, cash_flow_report = load_financial_reports(report_file_path)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 盈利能力
    yslr_path = os.path.join(save_dir, "yslr.png") if save_dir is not None else None
    llsyl_path = os.path.join(save_dir, "llsyl.png") if save_dir is not None else None
    mgzb_path = os.path.join(save_dir, "mgzb.png") if save_dir is not None else None
    plot_line_chart(profit_report, ["营业总收入", "净利润"], "营业收入与净利润", with_increment=True, save_path=yslr_path)

    items = [item for item in ["毛利率(%)", "净利率(%)", "加权净资产收益率(%)"] if item in main_stats_report.index]
    if items:
        plot_line_chart(main_stats_report, items, "利率与收益率", ylabel="比率(%)", with_increment=True, save_path=llsyl_path)

    items = [item for item in ["基本每股收益(元)", "扣非每股收益(元)", "稀释每股收益(元)"] if item in main_stats_report.index]
    if items:
        plot_line_chart(main_stats_report, items, "每股指标", ylabel="元", with_increment=True, save_path=mgzb_path)

    # 营收质量
    yszl_path = os.path.join(save_dir, "yszl.png") if save_dir is not None else None
    items = []
    if "应收票据及应收账款" in capital_debt_report.index:
        main_stats_report.loc["应收/营收"] = capital_debt_report.loc["应收票据及应收账款"] / profit_report.loc["营业总收入"] * 100
        items.append("应收/营收")
    if "销售现金流/营业收入" in main_stats_report.index:
        main_stats_report.loc["销售现金流/营收"] = main_stats_report.loc["销售现金流/营业收入"] * 100
        items.append("销售现金流/营收")
    if "经营现金流/营业收入" in main_stats_report.index:
        main_stats_report.loc["经营现金流/营收"] = main_stats_report.loc["经营现金流/营业收入"] * 100
        items.append("经营现金流/营收")
    plot_line_chart(main_stats_report, items, "营收质量指标", ylabel="比率(%)", with_increment=True, save_path=yszl_path)

    # 现金流
    xjll_path = os.path.join(save_dir, "xjll.png") if save_dir is not None else None
    cash_flow_report.loc["自由现金流"] = cash_flow_report.loc["经营活动产生的现金流量净额"] - cash_flow_report.loc["购建固定资产、无形资产和其他长期资产支付的现金"]
    mapping = {"经营活动产生的现金流量净额": "经营现金流", "投资活动产生的现金流量净额": "投资现金流", "筹资活动产生的现金流量净额": "筹资现金流", "期末现金及现金等价物余额": "期末现金流"}
    cash_flow_report = cash_flow_report.rename(index=lambda x: mapping.get(x, x))
    plot_line_chart(cash_flow_report, ["经营现金流", "投资现金流", "筹资现金流", "期末现金流", "自由现金流"], "现金流", with_increment=True, save_path=xjll_path)

    # 上下游地位指标
    items = []
    if "应付账款" in capital_debt_report.index:
        main_stats_report.loc["应付/营收"] = capital_debt_report.loc["应付账款"] / profit_report.loc["营业总收入"] * 100
        items.append("应付/营收")
    if "预付款项" in capital_debt_report.index:
        main_stats_report.loc["预付/营收"] = capital_debt_report.loc["预付款项"] / profit_report.loc["营业总收入"] * 100
        items.append("预付/营收")
    if "预收款项" in capital_debt_report.index:
        main_stats_report.loc["预收/营收"] = capital_debt_report.loc["预收款项"] / profit_report.loc["营业总收入"] * 100
        items.append("预收/营收")
    if items:
        yfyfys_path = os.path.join(save_dir, "yfyfys.png") if save_dir is not None else None
        plot_line_chart(main_stats_report, items, "应付预付预收比重", ylabel="比率(%)", with_increment=True, save_path=yfyfys_path)

    # 效率指标
    if "存货" in capital_debt_report.index:
        main_stats_report.loc["存货/营收"] = capital_debt_report.loc["存货"] / profit_report.loc["营业总收入"] * 100
        chys_path = os.path.join(save_dir, "chys.png") if save_dir is not None else None
        plot_double_y_chart(main_stats_report, "存货/营收", "存货/营收变化", ylabel="比率(%)", save_path=chys_path)
    zzts_path = os.path.join(save_dir, "zzts.png") if save_dir is not None else None
    items = []
    if "应收账款周转天数(天)" in main_stats_report.index:
        items.append("应收账款周转天数(天)")
    if "存货周转天数(天)" in main_stats_report.index:
        items.append("存货周转天数(天)")
    plot_line_chart(main_stats_report, items, "应收、存货周转天数", ylabel="天", with_increment=True, save_path=zzts_path)

    # 期间费用
    if "销售费用" in profit_report.index:
        qjfy_path = os.path.join(save_dir, "qjfy.png") if save_dir is not None else None
        qjfybl_path = os.path.join(save_dir, "qjfybl.png") if save_dir is not None else None
        plot_line_chart(profit_report, ["销售费用", "管理费用", "财务费用"], "期间费用", with_increment=True, save_path=qjfy_path)
        profit_report.loc["期间费用/营收"] = (profit_report.loc["销售费用"] + profit_report.loc["管理费用"] + profit_report.loc["财务费用"]) / profit_report.loc["营业总收入"] * 100
        plot_line_chart(profit_report, ["期间费用/营收"], "期间费用/营收", ylabel="比率(%)", with_increment=True, save_path=qjfybl_path)

    # 资产与负债

    zcgc_path = os.path.join(save_dir, "zcgc.png") if save_dir is not None else None
    recent_report = capital_debt_report[capital_debt_report.columns[-1]]
    if "流动资产合计" in recent_report.index:
        recent_report.loc["其他流动资产合计"] = (recent_report.loc["流动资产合计"] - recent_report.loc["货币资金"] - 
            recent_report.loc["应收票据及应收账款"] - recent_report["存货"])
        recent_report.loc["其他非流动资产合计"] = recent_report.loc["非流动资产合计"] - recent_report.loc["固定资产"]
        plot_pie_chart(recent_report.loc[["货币资金", "应收票据及应收账款", "存货", "其他流动资产合计", "固定资产", "其他非流动资产合计"]], ["货币资金", "应收票据及应收账款", 
            "存货", "其他流动资产合计", "固定资产", "其他非流动资产合计"], "资产构成", zcgc_path)

    fzgc_path = os.path.join(save_dir, "fzgc.png") if save_dir is not None else None
    if "短期借款" in recent_report.index and "流动负债合计" in recent_report.index:
        recent_report.loc["其他流动负债合计"] = recent_report.loc["流动负债合计"] - recent_report.loc["短期借款"] - recent_report.loc["应付票据及应付账款"]
        plot_pie_chart(recent_report.loc[["短期借款", "应付票据及应付账款", "其他流动负债合计", "非流动负债合计"]], ["短期借款", "应付票据及应付账款", "其他流动负债合计", "非流动负债合计"],
            "负债构成", fzgc_path)

    gdqy_path = os.path.join(save_dir, "gdqy.png") if save_dir is not None else None
    plot_double_y_chart(capital_debt_report, "归属于母公司股东权益合计", "股东权益变化", save_path=gdqy_path)

    # 风险指标
    zcfzl_path = os.path.join(save_dir, "zcfzl.png") if save_dir is not None else None
    capital_debt_report.loc["资产负债率"] = capital_debt_report.loc["负债合计"] / capital_debt_report.loc["资产总计"] * 100
    plot_double_y_chart(capital_debt_report, "资产负债率", "资产负债率变化", ylabel="比率(%)", save_path=zcfzl_path)
    if "速动比率" in main_stats_report.index:
        sdldbl_path = os.path.join(save_dir, "sdldbl.png") if save_dir is not None else None
        plot_line_chart(main_stats_report, ["速动比率", "流动比率"], "速动/流动比率", "比率", with_increment=True, save_path=sdldbl_path)


def plot_reports_comparison(stocks, report_file_paths, save_dir=None):
    main_stats_reports, profit_reports, cash_flow_reports, capital_debt_reports = load_reports_list(report_file_paths)

    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 盈利能力对比
    ysdb_path = os.path.join(save_dir, "ysdb.png") if save_dir is not None else None
    data = pd.DataFrame.from_records([reports.loc["营业总收入"] for reports in profit_reports], index=stocks)
    plot_adjacent_bar_chart(data, stocks, stocks, title="营收对比", save_path=ysdb_path)

    lrdb_path = os.path.join(save_dir, "lrdb.png") if save_dir is not None else None
    data = pd.DataFrame.from_records([reports.loc["净利润"] for reports in profit_reports], index=stocks)
    plot_adjacent_bar_chart(data, stocks, stocks, title="利润对比", save_path=lrdb_path)

    mldb_path = os.path.join(save_dir, "mldb.png") if save_dir is not None else None
    data = pd.DataFrame.from_records([reports.loc["毛利率(%)"] for reports in main_stats_reports], index=stocks)
    plot_adjacent_bar_chart(data, stocks, stocks, title="毛利率对比", save_path=mldb_path, ylabel="比率(%)")

    jldb_path = os.path.join(save_dir, "jldb.png") if save_dir is not None else None
    data = pd.DataFrame.from_records([reports.loc["净利率(%)"] for reports in main_stats_reports], index=stocks)
    plot_adjacent_bar_chart(data, stocks, stocks, title="净利率对比", save_path=jldb_path, ylabel="比率(%)")

    roedb_path = os.path.join(save_dir, "roedb.png") if save_dir is not None else None
    data = pd.DataFrame.from_records([reports.loc["加权净资产收益率(%)"] for reports in main_stats_reports], index=stocks)
    plot_adjacent_bar_chart(data, stocks, stocks, title="加权净资产收益率对比", save_path=roedb_path, ylabel="比率(%)")

    # 现金流对比
    jyxjldb_path = os.path.join(save_dir, "jyxjldb.png") if save_dir is not None else None
    data = pd.DataFrame.from_records([reports.loc["经营活动产生的现金流量净额"] for reports in cash_flow_reports], index=stocks)
    plot_adjacent_bar_chart(data, stocks, stocks, title="经营现金流对比", save_path=jyxjldb_path)
    for cash_flow_report in cash_flow_reports:
        cash_flow_report.loc["自由现金流"] = cash_flow_report.loc["经营活动产生的现金流量净额"] - cash_flow_report.loc["购建固定资产、无形资产和其他长期资产支付的现金"]
    zyxjldb_path = os.path.join(save_dir, "zyxjldb.png") if save_dir is not None else None
    data = pd.DataFrame.from_records([reports.loc["自由现金流"] for reports in cash_flow_reports], index=stocks)
    plot_adjacent_bar_chart(data, stocks, stocks, title="自由现金流对比", save_path=zyxjldb_path)

    # 资产负债对比
    for capital_debt_report in capital_debt_reports:
        capital_debt_report.loc["资产负债率"] = capital_debt_report.loc["负债合计"] / capital_debt_report.loc["资产总计"] * 100
    fzldb_path = os.path.join(save_dir, "fzldb.png") if save_dir is not None else None
    data = pd.DataFrame.from_records([reports.loc["资产负债率"] for reports in capital_debt_reports], index=stocks)
    plot_adjacent_bar_chart(data, stocks, stocks, title="资产负债率对比", save_path=fzldb_path, ylabel="比率(%)")

    ldbldb_path = os.path.join(save_dir, "ldbldb.png") if save_dir is not None else None
    data = pd.DataFrame.from_records([reports.loc["流动比率"] for reports in main_stats_reports], index=stocks)
    plot_adjacent_bar_chart(data, stocks, stocks, title="流动比率对比", save_path=ldbldb_path, ylabel="比率")


if __name__ == "__main__":
    # plot_reports("../data/600009/财务报表.xlsx", "../static/600009/")
    # plot_reports_comparison(["格力电器", "美的集团"], ["../data/000651/财务报表.xlsx", "../data/000333/财务报表.xlsx"])
    # plot_pie_chart([1,1,40,20], ["1", "2", "3", "4"], "示例")
    # import pandas as pd
    # data = pd.read_excel("data/600009/上海机场营业数据.xlsx", index_col=0)
    # plot_adjacent_bar_chart(data, ["飞机起降架次", "国内飞机起降架次", "国际飞机起降架次"], ["飞机起降架次", "国内起降架次", "国际起降架次"], "飞机起降架次", ylabel="起降架次", save_path="static/600009/jiaci.png")
    # data.loc["航空性收入"] = data.loc["航空性收入"] / 100000000
    # data.loc["非航空性收入"] = data.loc["非航空性收入"] / 100000000
    # plot_adjacent_bar_chart(data, ["航空性收入", "非航空性收入"], ["航空性收入", "非航空性收入"], "航空与非航收入", save_path="static/600009/shouru.png")

    profit_reports, asset_reports, cash_reports = load_financial_reports("000001")
    plot_overlap_bar_chart(profit_reports, ["营业收入", "净利润"], ["营业收入", "净利润"], "营业收入与净利润", save_path="yingshou.png", with_increment=True)
