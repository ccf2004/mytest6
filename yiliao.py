import streamlit as st
import pickle
import pandas as pd

#1. 页面配置（必须放在最外层，且是第一个 Streamlit 命令）
st.set_page_config(
    page_title="医疗费用预测",
    page_icon="🏥",  # 可选：添加一个图标，更直观
    layout="wide"    # 可选：宽布局，体验更好
)

#2. 简介页面函数
def introduce_page():
    """当选择简介页面时，将呈现该函数的内容"""
    st.write("# 欢迎使用!")  # 修复：# 后加空格
    
    st.sidebar.success("单击预测医疗费用")
    
    st.markdown(
        """
        # 医疗费用预测应用
        这个应用利用机器学习模型来预测医疗费用，为保险公司的保险定价提供参考。
        
        ## 背景介绍
        - 开发目标：帮助保险公司合理定价保险产品，控制风险。
        - 模型算法：利用随机森林回归算法训练医疗费用预测模型。
        
        ## 使用指南
        - 输入准确完整的被保险人信息，可以得到更准确的费用预测。
        - 预测结果可以作为保险定价的重要参考，但需审慎决策。
        - 有任何问题欢迎联系我们的技术支持。

        技术支持：email：support@example.com
        """
    )

#  3. 预测页面函数 
def predict_page():
    """当选择预测费用页面时，将呈现该函数的内容"""
    st.markdown(
        """
        ## 使用说明
        这个应用利用机器学习模型来预测医疗费用，为保险公司的保险定价提供参考。
        - **输入信息**：在下面输入被保险人的个人信息，疾病信息等。
        - **费用预测**：应用会预测被保险人的未来医疗费用支出。
        """
    )

    # 运用表单和表单提交按钮
    with st.form('user_inputs'):
        age = st.number_input('年龄', min_value=0, max_value=120)  # 增加最大值，更合理
        sex = st.radio('性别', options=['男性', '女性'])
        # -------------------------- 关键修改：添加step=0.1，匹配format的1位小数 --------------------------
        bmi = st.number_input(
            'BMI', 
            min_value=0.0, 
            max_value=100.0, 
            format="%.1f",
            step=0.1  # 点击+/-时每次增减0.1，和显示格式一致
        )  
        
        children = st.number_input("子女数量", step=1, min_value=0, max_value=20)  # 增加最大值
        smoke = st.radio("是否吸烟", ("是", "否"))
        region = st.selectbox('区域', ('东南部', '西南部', '东北部', '西北部'))
        submitted = st.form_submit_button('预测费用')

    if submitted:
        # -------------------------- 修复：变量初始化（避免未定义错误）--------------------------
        sex_female, sex_male = 0, 0
        if sex == '女性':
            sex_female = 1
        else:  # 男性（else 更简洁，避免漏判）
            sex_male = 1

        # 修复：smoke 变量在外部初始化，避免作用域问题
        smoke_yes, smoke_no = 0, 0
        if smoke == '是':
            smoke_yes = 1
        else:  # 否
            smoke_no = 1

        region_northeast, region_southeast, region_northwest, region_southwest = 0, 0, 0, 0
        if region == '东北部':
            region_northeast = 1
        elif region == '东南部':
            region_southeast = 1
        elif region == '西北部':
            region_northwest = 1
        elif region == '西南部':
            region_southwest = 1

        # 格式化输入数据
        format_data = [
            age, bmi, children, sex_female, sex_male,
            smoke_no, smoke_yes, region_northeast,
            region_southeast, region_northwest, region_southwest
        ]

        # -------------------------- 加载模型并预测 --------------------------
        try:
            # 加载保存的随机森林模型（确保 rfr_model.pkl 和代码在同一文件夹）
            with open('rfr_model.pkl', 'rb') as f:
                rfr_model = pickle.load(f)

            # 转换为 DataFrame（匹配模型训练时的特征名）
            format_data_df = pd.DataFrame(
                data=[format_data],
                columns=rfr_model.feature_names_in_
            )

            # 预测并输出结果
            predict_result = rfr_model.predict(format_data_df)[0]
            st.success(f'✅ 根据您输入的数据，预测该客户的医疗费用是：{round(predict_result, 2)} 元')

        except FileNotFoundError:
            st.error("❌ 未找到模型文件 'rfr_model.pkl'！请确保模型文件与代码在同一文件夹。")
        except Exception as e:
            st.error(f"❌ 预测出错：{str(e)}")

    st.markdown("---")  # 分割线，更美观
    st.write("技术支持：email：support@example.com")

# 4. 侧边栏导航（放在最外层，函数之外）
nav = st.sidebar.radio("导航", ["简介", "预测医疗费用"]) 

# 根据选择的导航，展示对应页面
if nav == "简介":
    introduce_page()
else:
    predict_page()
