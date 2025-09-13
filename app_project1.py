import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import squarify
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from datetime import datetime
from dataclasses import dataclass
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import RobustScaler
from scipy.cluster.hierarchy import linkage, dendrogram

df_transactions = pd.read_csv('Transactions.csv')
df_products = pd.read_csv('Products_with_Categories.csv')

df_transactions['Converted_Date'] = df_transactions['Date'].apply(lambda x: datetime.strptime(x, '%d-%m-%Y').date())

max_date = df_transactions['Converted_Date'].max()

df_merged = pd.merge(df_products, df_transactions, on='productId')
df_merged['total_spent'] = df_merged['price'] * df_merged['items']

# ======================== User Interface ======================== #

def page_business_problem() -> st.Page:
  st.image('general_store_banner.jpg', width='content')
  st.markdown('''
    # VẤN ĐỀ KINH DOANH
    #### Cửa hàng X là một cửa hàng :blue[tạp hóa].
    \b          
    #### Cửa hàng X chuyên cung cấp các sản phẩm thiết yếu hằng ngày:
    #### - Rau củ, trái cây, thịt cá.
    #### - Sữa, trứng, sữa chua.
    #### - Bánh mì, bánh ngọt.
    #### - Nước giải khát và thực phẩm chế biến sẵn.
    \b
    # KHÓ KHĂN HIỆN TẠI          
    #### - Khó xác định khách hàng nào thực sự mang lại giá trị cao.  
    #### - Chưa có công cụ theo dõi xu hướng mua sắm theo thời gian.  
    #### - Chưa có cách giữ chân khách hàng trung thành và thu hút lại khách hàng cũ.          
    \b
    # CÂU HỎI
    #### - Sản phẩm nào được mua nhiều nhất?  
    #### - Ai là khách hàng trung thành, ai có nguy cơ rời bỏ?  
    #### - Làm sao để tăng doanh thu bằng cách chăm sóc đúng nhóm khách hàng?
    \b 
    # MỤC TIÊU
    #### - Phân tích dữ liệu giao dịch để hiểu rõ hành vi mua sắm.  
    #### - Xây dựng phân khúc khách hàng và dự đoán khả năng rời bỏ.  
    #### - Đưa ra khuyến nghị giúp giữ chân khách hàng và tối ưu doanh thu.           
  ''')

def page_eda() -> st.Page:  
  min_date_ = df_merged['Converted_Date'].min()
  max_date_ = df_merged['Converted_Date'].max()

  st.markdown('# TỔNG QUAN DỮ LIỆU')
  
  # --------- Product Overview --------- #

  products_overview_ = df_products.groupby('Category')['productName'].nunique().reset_index()
  products_overview_.columns = ['Category', 'Number of Products']
  
  st.markdown(f'''
    \b
    #### Cửa hàng hiện cung cấp :blue[{df_products['productName'].nunique()}] mặt hàng, chia thành :blue[{df_products['Category'].nunique()}] danh mục khác nhau.
    #### Số lượng mặt hàng theo từng danh mục:
  ''')

  st.dataframe(products_overview_)

  # --------- Monthly Purchase --------- #

  monthly_purchase_ = (
    df_merged.groupby(df_merged['Converted_Date'])['items']
    .sum()
    .reset_index()
  )

  st.markdown(f'''
    \b 
    #### Số lượng sản phẩm được mua theo từng tháng:
  ''')

  monthly_purchase_start_, monthly_purchase_end_ = st.slider(
    key='monthly_purchase_range',
    label='monthly_purchase_range',
    label_visibility='hidden',
    min_value=min_date_,
    max_value=max_date_,
    value=(min_date_, max_date_),
    format='MM/YYYY'
  )

  filtered_purchase_ = monthly_purchase_[
    (monthly_purchase_['Converted_Date'] >= monthly_purchase_start_) &
    (monthly_purchase_['Converted_Date'] <= monthly_purchase_end_)
  ]

  st.line_chart(filtered_purchase_.set_index('Converted_Date'), color='#00ff00')

  # --------- Top 10 Products --------- #

  st.markdown(f'''
    \b 
    #### 10 sản phẩm được khách hàng mua nhiều nhất:
  ''')
  
  categories_ = df_products['Category'].unique().tolist()
  categories_.insert(0, 'All')

  top_10_category_option_ = st.selectbox(
    label='Chọn danh mục',
    options=categories_
  )

  top_10_products_ = (
    df_merged.groupby('productName')['items']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
  )

  top_10_products_by_category_ = (
    df_merged[df_merged['Category'] == top_10_category_option_].groupby('productName')['items']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
  )

  filtered_top_10_products_ = (
    top_10_products_ 
    if (top_10_category_option_ == 'All') 
    else top_10_products_by_category_
  ) 

  st.bar_chart(filtered_top_10_products_.set_index('productName'))

  # --------- Monthly Revenue --------- #

  df_merged['total_spent'] = df_merged['price'] * df_merged['items']
  monthly_revenue_ = (
    df_merged.groupby(df_merged['Converted_Date'])['total_spent']
    .sum()
    .reset_index()
  )

  st.markdown(f'''
    \b 
    #### Tổng số tiền khách hàng đã chi tiêu theo từng tháng:
  ''')
  
  monthly_revenue_start_, monthly_revenue_end_ = st.slider(
    key='monthly_revenue_range',
    label='monthly_revenue_range',
    label_visibility='hidden',
    min_value=min_date_,
    max_value=max_date_,
    value=(min_date_, max_date_),
    format='MM/YYYY'
  )
  
  filtered_revenue_ = monthly_revenue_[
    (monthly_revenue_['Converted_Date'] >= monthly_revenue_start_) &
    (monthly_revenue_['Converted_Date'] <= monthly_revenue_end_)
  ]

  st.line_chart(filtered_revenue_.set_index('Converted_Date'), color='#ffaa00')

def page_analysis_pred() -> st.Page:
  @dataclass
  class CustomerClass:
    label: str
    recency: int
    frequency: int
    monetary: int
    colour: str = '#ffffff'
  
  f_recency = lambda x: (max_date - x.max()).days
  f_frequency = lambda x: len(x.unique())
  f_monetary = lambda x : round(sum(x), 2)
  
  df_rfm = df_merged.groupby('Member_number').agg({
      'Converted_Date': f_recency,
      'productId': f_frequency,
      'total_spent': f_monetary
  })

  df_rfm.columns = ['Recency', 'Frequency', 'Monetary']
  df_rfm = df_rfm.sort_values('Monetary', ascending=False)

  r_labels = range(4, 0, -1)
  f_labels = range(1, 5)
  m_labels = range(1, 5)

  r_groups = pd.qcut(df_rfm['Recency'].rank(method='first'), q=4, labels=r_labels)
  f_groups = pd.qcut(df_rfm['Frequency'].rank(method='first'), q=4, labels=f_labels)
  m_groups = pd.qcut(df_rfm['Monetary'].rank(method='first'), q=4, labels=m_labels)

  def join_rfm(x): return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))

  df_rfm = df_rfm.assign(R = r_groups.values, F = f_groups.values,  M = m_groups.values)
  df_rfm['RFM_Segment'] = df_rfm.apply(join_rfm, axis=1)
  df_rfm['RFM_Score'] = df_rfm[['R','F','M']].sum(axis=1)
  
  # ----------- Interface ----------- #

  st.markdown('''
    # PHÂN TÍCH & DỰ ĐOÁN
    \b
  ''')

  options_ = [
    'Manual', 
    'KMeans', 
    'Hierarchical']

  selected_option_ = st.selectbox(
    label='Chọn phương pháp',
    options=options_
  )

  with st.container(border=True):
    # -------------- Manual Segmentation -------------- #
    if selected_option_ == 'Manual':
      if 'customer_classes' not in st.session_state:
        st.session_state.customer_classes = [
          CustomerClass('VIP', 4, 4, 4, '#ffd700'),
          CustomerClass('NEW', 4, 2, 2, '#1e90ff'),
          CustomerClass('LOYAL', 3, 4, 3, '#32cd32'),
          CustomerClass('BIG SPENDER', 2, 2, 4, '#ff4500'),
          CustomerClass('POTENTIAL', 3, 3, 2, '#9370db'),
          CustomerClass('AT RISK', 2, 2, 2, '#ff69b4'),
          CustomerClass('LOST', 1, 1, 1, '#a9a9a9') 
        ]

      if 'vip_manual' not in st.session_state:
        st.session_state.vip_manual = {
          'Segment': 'VIP (Manual)',
          'RecencyMean': 0,
          'FrequencyMean': 0,
          'MonetaryMean': 0,
          'Count': 0,
          'Percent': 0
        }

      if 'df_rfm' not in st.session_state:
        st.session_state.df_rfm = pd.DataFrame()

      customer_classes_ = st.session_state.customer_classes
      class_options_ = [class_.label for class_ in customer_classes_]

      with st.expander(label='Thêm Phân khúc Khách hàng'):
        with st.form(key='add_new_customer_class', clear_on_submit=True, border=False):
          new_label_ = st.text_input('Tên')
          r_val_ = st.slider('Recency', 1, 4, 2)
          f_val_ = st.slider('Frequency', 1, 4, 2)
          m_val_ = st.slider('Monetary', 1, 4, 2)
          colour_ = st.color_picker(label='Màu', value='#ffffff')

          submitted_ = st.form_submit_button('Thêm')
          if submitted_:
            st.session_state.customer_classes.append(CustomerClass(new_label_, r_val_, f_val_, m_val_, colour_))
            
            st.success(f'Đã thêm: {new_label_}')
            st.rerun()

      with st.expander(label='Xóa Phân khúc Khách hàng'):
        classes_to_delete_ = st.multiselect(
          key='classes_to_delete',
          label='selected_labels',
          label_visibility='collapsed',
          options=[class_.label for class_ in customer_classes_]
        )

        if st.button('Xoá'):
          st.session_state.customer_classes = [class_ for class_ in customer_classes_ if class_.label not in classes_to_delete_]
          
          st.success(f"Đã xoá: {', '.join(classes_to_delete_)}")
          st.rerun()

      selected_labels_ = st.multiselect(
        key='selected_labels',
        label='selected_labels',
        label_visibility='collapsed',
        options=class_options_,
        default=class_options_
      )
      selected_classes_ = [class_ for class_ in customer_classes_ if class_.label in selected_labels_]

      def rfm_tiering(df):
        for class_ in selected_classes_:
          if (df['R'] == class_.recency) and (df['F'] == class_.frequency) and (df['M'] == class_.monetary):
            return class_.label  

      df_rfm['RFM_Tier'] = df_rfm.apply(rfm_tiering, axis=1)

      df_rfm_agg = df_rfm.groupby('RFM_Tier').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']}
      ).round(0)  

      df_rfm_agg.columns = df_rfm_agg.columns.droplevel()
      df_rfm_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
      df_rfm_agg['Percent'] = round((df_rfm_agg['Count'] / df_rfm_agg.Count.sum()) * 100, 2)

      df_rfm_agg = df_rfm_agg.reset_index()

      graph_row_ = st.columns(2)

      # TreeMap
      with graph_row_[0].container(border=True):
        st.markdown('#### TreeMap')

        fig_, ax_ = plt.subplots()
        
        colour_coding = {
          class_.label: class_.colour for class_ in customer_classes_ if class_.label in df_rfm_agg['RFM_Tier'].unique()
        }

        squarify.plot(
          sizes = df_rfm_agg['Count'],
          text_kwargs = {'fontsize': 5, 'weight': 'regular', 'fontname': 'sans serif'},
          label = ['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*df_rfm_agg.iloc[i]) for i in range(0, len(df_rfm_agg))],
          color=[colour_coding.get(label_, '#ffffff') for label_ in df_rfm_agg['RFM_Tier']],
          alpha = 0.5
        )

        plt.axis('off')
        st.pyplot(fig_)
        st.session_state['treemap_manual'] = fig_

      # Scatter
      with graph_row_[1].container(border=True):
        st.markdown('#### Scatter')
        
        colour_coding = {
          class_.label: class_.colour for class_ in customer_classes_ if class_.label in df_rfm_agg['RFM_Tier'].unique()
        }

        fig_ = px.scatter(
          df_rfm_agg,
          x='RecencyMean',
          y='MonetaryMean',
          size='FrequencyMean',
          color='RFM_Tier',
          color_discrete_map=colour_coding,
          hover_name='RFM_Tier',
          hover_data=['Count', 'Percent']
        )

        st.plotly_chart(fig_)
        st.session_state['scatter_manual'] = fig_

      # Save VIP for manual segmentation 
      vip_ = df_rfm_agg[df_rfm_agg['RFM_Tier'] == 'VIP']
      if not vip_.empty:
        row_ = vip_.iloc[0]
        st.session_state.vip_manual = {
          'Segment': 'VIP (Manual)',
          'RecencyMean': row_['RecencyMean'],
          'FrequencyMean': row_['FrequencyMean'],
          'MonetaryMean': row_['MonetaryMean'],
          'Count': row_['Count'],
          'Percent': row_['Percent']
        }

      st.session_state.df_rfm = df_rfm

    # -------------- KMeans Segmentation -------------- #
    elif selected_option_ == 'KMeans':
      if 'vip_kmeans' not in st.session_state:
        st.session_state.vip_kmeans = {
          'Segment': 'VIP (KMeans)',
          'RecencyMean': 0,
          'FrequencyMean': 0,
          'MonetaryMean': 0,
          'Count': 0,
          'Percent': 0
        }

      if 'df_rfm' not in st.session_state:
        st.session_state.df_rfm_km = pd.DataFrame()

      df_rfm_km = df_rfm[['Recency','Frequency','Monetary']]

      scaler_ = RobustScaler()
      df_rfm_km_scaled = scaler_.fit_transform(df_rfm_km)

      graph_row_ = st.columns(2)

      # Elbow Method
      with graph_row_[0].container(border=False):
        sse_ = {}
        
        for k_ in range(2, 10):
          kmeans_ = KMeans(n_clusters=k_, random_state=42)
          kmeans_.fit(df_rfm_km_scaled)
          
          sse_[k_] = kmeans_.inertia_

        st.markdown('### Elbow Method')
        
        fig_, ax_ = plt.subplots(figsize=(10, 6))
        ax_.plot(list(sse_.keys()), list(sse_.values()), marker='o', linestyle='-', color='steelblue')
        ax_.set_xlabel('k')
        ax_.set_ylabel('SSE')

        st.pyplot(fig_)

      # Silhoutte Score
      with graph_row_[1].container(border=False):
        silhouette_scores_ = []

        for k_ in range(2, 10):
          kmeans_ = KMeans(n_clusters=k_, random_state=42)
          pred_ = kmeans_.fit_predict(df_rfm_km_scaled)
          score_ = silhouette_score(df_rfm_km_scaled, pred_)
          
          silhouette_scores_.append(score_)

        st.markdown('### Silhoutte Scores')

        fig_, ax_ = plt.subplots(figsize=(10, 6))
        ax_.plot(range(2, 10), silhouette_scores_, marker='o', linestyle='-', color='steelblue')  
        ax_.set_xlabel('k')
        ax_.set_ylabel('Silhouette Scores')

        st.pyplot(fig_)      

      st.markdown('\b')

      if 'kmeans_selected_k' not in st.session_state:
        st.session_state.kmeans_selected_k = 4

      selected_k_ = st.select_slider(
        label='Select K',
        options=range(2, 10),
        value=4
      )
      
      st.session_state.kmeans_selected_k = selected_k_

      model_ = KMeans(n_clusters=selected_k_, random_state=42)
      model_.fit(df_rfm_km_scaled)

      df_rfm_km['Cluster'] = model_.labels_

      df_rfm_agg2 = df_rfm_km.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']}
      ).round(0)

      df_rfm_agg2.columns = df_rfm_agg2.columns.droplevel()
      df_rfm_agg2.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
      df_rfm_agg2['Percent'] = round((df_rfm_agg2['Count'] / df_rfm_agg2.Count.sum()) * 100, 2)

      df_rfm_agg2 = df_rfm_agg2.reset_index()

      df_rfm_agg2['Cluster'] = 'Cluster '+ df_rfm_agg2['Cluster'].astype('str')

      st.markdown('\b')

      graph_row_2_ = st.columns(2)

      # Tree Map
      with graph_row_2_[0].container(border=True):
        st.markdown('### TreeMap')
        
        fig_, ax_ = plt.subplots(figsize=(10, 6))

        cluster_labels_ = df_rfm_agg2['Cluster'].tolist()
        colours_ = plt.cm.Set3.colors[:len(cluster_labels_)]

        squarify.plot(
          sizes=df_rfm_agg2['Count'],
          label = ['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*df_rfm_agg2.iloc[i]) for i in range(0, len(df_rfm_agg2))],
          color=colours_,
          text_kwargs={'fontsize': 5, 'weight': 'regular', 'fontname': 'sans serif'},
          alpha=0.6
        )

        plt.axis('off')
        st.pyplot(fig_)
        st.session_state['treemap_kmeans'] = fig_

      # Scatter
      with graph_row_2_[1].container(border=True):
        st.markdown('### Scatter')

        fig_ = px.scatter(
          df_rfm_agg2,
          x='RecencyMean',
          y='MonetaryMean',
          size='FrequencyMean',
          color='Cluster',
          hover_name='Cluster',
          hover_data=['Count', 'Percent']
        )

        st.plotly_chart(fig_)
        st.session_state['scatter_kmeans'] = fig_

      # Save VIP for KMeans clustering
      df_rfm_agg2_ranked = df_rfm_agg2.sort_values(
        by=['FrequencyMean', 'MonetaryMean', 'RecencyMean'],
        ascending=[False, False, True]
      )

      vip_ = df_rfm_agg2_ranked.iloc[0]
      if not vip_.empty:
        st.session_state.vip_kmeans = {
          'Segment': f"{vip_['Cluster']} (KMeans - VIP)",
          'RecencyMean': vip_['RecencyMean'],
          'FrequencyMean': vip_['FrequencyMean'],
          'MonetaryMean': vip_['MonetaryMean'],
          'Count': vip_['Count'],
          'Percent': vip_['Percent']
        }

      st.session_state.df_rfm_km = df_rfm_km

    # -------------- Hierarchical Segmentation -------------- #
    elif selected_option_ == 'Hierarchical':
      if 'vip_hierarchical' not in st.session_state:
        st.session_state.vip_hierarchical = {
          'Segment': 'VIP (Hierarchical)',
          'RecencyMean': 0,
          'FrequencyMean': 0,
          'MonetaryMean': 0,
          'Count': 0,
          'Percent': 0
        }

      if 'df_rfm' not in st.session_state:
        st.session_state.df_rfm_hc = pd.DataFrame()
      
      df_rfm_hc = df_rfm[['Recency','Frequency','Monetary']]

      scaler_ = RobustScaler()
      df_rfm_hc_scaled = scaler_.fit_transform(df_rfm_hc)

      graph_row_ = st.columns(2, gap='medium')

      # First Column - Dendrogram & Silhoutte Scores
      with graph_row_[0].container(border=False, height=800):
        # Dendrogram
        st.markdown('### Dendrogram')
        
        hc_linkage_ = linkage(df_rfm_hc_scaled, method='ward')
        fig_, ax_ = plt.subplots(figsize=(10, 6))

        dendrogram(hc_linkage_, leaf_rotation=90, leaf_font_size=12, no_labels=True)

        ax_.set_ylabel('Distance')

        st.pyplot(fig_)

        # Silhoutte Scores
        silhouette_scores_ = []

        for k_ in range(2, 10):
          ac_ = AgglomerativeClustering(n_clusters=k_)
          pred_ = ac_.fit_predict(df_rfm_hc_scaled)
          score_ = silhouette_score(df_rfm_hc_scaled, pred_)
          
          silhouette_scores_.append(score_)

        st.markdown('\b')
        st.markdown('### Silhoutte Scores')

        fig_, ax_ = plt.subplots(figsize=(10, 6))
        ax_.plot(range(2, 10), silhouette_scores_, marker='o', linestyle='-', color='steelblue')  
        ax_.set_xlabel('k')
        ax_.set_ylabel('Silhouette Scores')

        st.pyplot(fig_)

      # Second column - Silhoutte Density
      with graph_row_[1].container(border=False, height=800):
        st.markdown('### Silhoutte Density')

        k_range_ = range(2, 10)
        silhouette_scores_ = []

        fig_, axes_ = plt.subplots(len(k_range_), 1, figsize=(8, len(k_range_) * 3))
        
        if len(k_range_) == 1:
          axes_ = [axes_]

        for idx_, k_ in enumerate(k_range_):
          axis_ = axes_[idx_]

          model_ = AgglomerativeClustering(n_clusters=k_)
          pred_ = model_.fit_predict(df_rfm_hc_scaled)

          silhouette_vals_ = silhouette_samples(df_rfm_hc_scaled, pred_)
          score_ = silhouette_score(df_rfm_hc_scaled, pred_)
          silhouette_scores_.append(score_)

          y_lower_ = 10
          
          for i_ in range(k_):
            cluster_vals_ = silhouette_vals_[pred_ == i_]
            cluster_vals_.sort()

            size_ = cluster_vals_.shape[0]
            y_upper_ = y_lower_ + size_

            colour_ = cm.nipy_spectral(float(i_) / k_)
            axis_.fill_betweenx(
              np.arange(y_lower_, y_upper_),
              0, cluster_vals_,
              facecolor=colour_, 
              edgecolor=colour_, 
              alpha=0.7
            )

            axis_.text(-.05, y_lower_ + .05 * size_, str(i_))
            y_lower_ = y_upper_ + 10

          axis_.axvline(x=score_, color='red', linestyle='--')
          axis_.set_xlim([-0.1, 1])
          axis_.set_ylim([0, len(df_rfm_hc_scaled) + (k_ + 1) * 10])
          axis_.set_xlabel('Silhouette Coefficient Values')
          axis_.set_ylabel('Clustered Samples')
          axis_.set_title(f'k = {k_} | Avg Silhouette Score = {score_:.4f}')

        fig_.tight_layout()
        st.pyplot(fig_)

      st.markdown('\b')

      if 'hierarchical_selected_k' not in st.session_state:
        st.session_state.hierarchical_selected_k = 4

      selected_k_ = st.select_slider(
        label='Select K',
        options=range(2, 10),
        value=4
      )

      st.session_state.hierarchical_selected_k = selected_k_

      model_ = AgglomerativeClustering(n_clusters=selected_k_)
      pred_ = model_.fit_predict(df_rfm_hc_scaled)

      df_rfm_hc['Cluster'] = pred_
      
      df_rfm_agg3 = df_rfm_hc.groupby('Cluster').agg({
        'Recency':'mean',
        'Frequency':'mean',
        'Monetary':['mean', 'count']
      }).round(2)

      df_rfm_agg3.columns = df_rfm_agg3.columns.droplevel()
      df_rfm_agg3.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
      df_rfm_agg3['Percent'] = round((df_rfm_agg3['Count'] / df_rfm_agg3.Count.sum()) * 100, 2)

      df_rfm_agg3 = df_rfm_agg3.reset_index()
      df_rfm_agg3['Cluster'] = 'Cluster '+ df_rfm_agg3['Cluster'].astype('str')

      graph_row_2_ = st.columns(2)

      with graph_row_2_[0].container(border=True):
        st.markdown('### TreeMap')
        
        fig_, ax_ = plt.subplots(figsize=(10, 6))

        cluster_labels_ = df_rfm_agg3['Cluster'].tolist()
        colours_ = plt.cm.Set3.colors[:len(cluster_labels_)]

        squarify.plot(
          sizes=df_rfm_agg3['Count'],
          label = ['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*df_rfm_agg3.iloc[i]) for i in range(0, len(df_rfm_agg3))],
          color=colours_,
          text_kwargs={'fontsize': 5, 'weight': 'regular', 'fontname': 'sans serif'},
          alpha=0.6
        )

        plt.axis('off')
        st.pyplot(fig_)
        st.session_state['treemap_hierarchical'] = fig_

      # Scatter
      with graph_row_2_[1].container(border=True):
        st.markdown('### Scatter')

        fig_ = px.scatter(
          df_rfm_agg3,
          x='RecencyMean',
          y='MonetaryMean',
          size='FrequencyMean',
          color='Cluster',
          hover_name='Cluster',
          hover_data=['Count', 'Percent']
        )

        st.plotly_chart(fig_)
        st.session_state['scatter_hierarchical'] = fig_

      # Save VIP for hierarchical clustering
      df_rfm_agg3_ranked = df_rfm_agg3.sort_values(
        by=['FrequencyMean', 'MonetaryMean', 'RecencyMean'],
        ascending=[False, False, True]
      )

      vip_ = df_rfm_agg3_ranked.iloc[0]
      if not vip_.empty:
        st.session_state.vip_hierarchical = {
          'Segment': f"{vip_['Cluster']} (Hierarchical - VIP)",
          'RecencyMean': vip_['RecencyMean'],
          'FrequencyMean': vip_['FrequencyMean'],
          'MonetaryMean': vip_['MonetaryMean'],
          'Count': vip_['Count'],
          'Percent': vip_['Percent']
        }

      st.session_state.df_rfm_hc = df_rfm_hc

def page_eval_report() -> st.Page:
  if 'selected_method' not in st.session_state:
    st.session_state.selected_method = 'Manual'
  
  st.markdown('''
    # ĐÁNH GIÁ & BÁO CÁO
    \b
  ''')

  # segment comparison
  with st.container(border=True):
    segment_row_ = st.columns(3, gap='medium')

    methods_ = ['manual', 'kmeans', 'hierarchical']
    titles_ = ['Manual Segments', 'KMeans Clusters', 'Agglomerative Clusters']

    for index_, method_ in enumerate(methods_):
      with segment_row_[index_].container(border=True):
        st.markdown(f'#### {titles_[index_]}')

        # TreeMap
        treemap_ = st.session_state.get(f'treemap_{method_}')
        if treemap_:
          st.pyplot(treemap_)

        # Scatter
        scatter_ = st.session_state.get(f'scatter_{method_}')
        if scatter_:
          st.plotly_chart(scatter_)

  # VIP table comparison
  df_vip = pd.DataFrame([
    st.session_state.get('vip_manual', {}),
    st.session_state.get('vip_kmeans', {}),
    st.session_state.get('vip_hierarchical', {})
  ])

  df_vip = df_vip.set_index('Segment')
  st.dataframe(
    df_vip.style
    .background_gradient(cmap='YlOrRd', subset=['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count', 'Percent'])
    .format(precision=2)
  )

  # method selection
  with st.container(border=True):
    st.markdown('### Chọn một Phương pháp')
    
    selection_row_ = st.columns(3)

    with selection_row_[0]:
      if st.button('Manual' + (' ✅' if st.session_state.selected_method == 'Manual' else ''), width='stretch'):
          st.session_state.selected_method = 'Manual'
          st.rerun()

    with selection_row_[1]:
      if st.button('KMeans' + (' ✅' if st.session_state.selected_method == 'KMeans' else ''), width='stretch'):
          st.session_state.selected_method = 'KMeans'
          st.rerun()

    with selection_row_[2]:
      if st.button('Hierarchical' + (' ✅' if st.session_state.selected_method == 'Hierarchical' else ''), width='stretch'):
          st.session_state.selected_method = 'Hierarchical'
          st.rerun()

def page_recommendation() -> st.Page:
  st.markdown('''
    # KHUYẾN NGHỊ
    \b
  ''')

  if 'selected_top_n' not in st.session_state:
    st.session_state.selected_top_n = 5 

  selected_method_ = st.session_state.selected_method

  df_rfm_ = st.session_state.df_rfm
  df_rfm_km_ = st.session_state.df_rfm_km
  df_rfm_hc_ = st.session_state.df_rfm_hc

  st.markdown(f'''
    #### Phương pháp đã chọn: :blue[{selected_method_}]
    \b
  ''')
  
  st.markdown(f'#### Sản phẩm được mua nhiều nhất theo Phân khúc')

  with st.container(border=True):
    top_n_ = st.number_input(
      label='Số sản phẩm', 
      min_value=1, 
      max_value=df_products['productName'].nunique(),
      value=5
    )

    if st.session_state.selected_top_n != top_n_:
      st.session_state.selected_top_n = top_n_
      st.rerun()

    top_n_product_ids_ = df_transactions.groupby('productId')['items'].sum().nlargest(st.session_state.selected_top_n).index.tolist()

    if selected_method_ == 'Manual':
      df_rfm_ = df_rfm_.reset_index()
      df_merged = df_transactions[df_transactions['productId'].isin(top_n_product_ids_)]
      df_merged = df_merged.merge(df_rfm_[['Member_number', 'RFM_Tier']], on='Member_number', how='left')
      df_merged = df_merged.merge(df_products[['productId', 'productName']], on='productId', how='left')

      purchase_counts_ = df_merged.groupby(['RFM_Tier', 'productName'])['items'].sum().reset_index()

      fig_ = px.bar(
        purchase_counts_,
        x='items',
        y='productName',
        color='RFM_Tier',
        orientation='h',
        title='Top Purchased Products by Segment',
        labels={'items': 'Total Items Purchased', 'productName': 'Product'},
        barmode='group',
        height=600
      )

      fig_.update_layout(yaxis={'categoryorder':'total ascending'})
      st.plotly_chart(fig_)

    elif selected_method_ == 'KMeans':
      df_rfm_km_ = df_rfm_km_.reset_index()
      df_merged = df_transactions[df_transactions['productId'].isin(top_n_product_ids_)]
      df_merged = df_merged.merge(df_rfm_km_[['Member_number', 'Cluster']], on='Member_number', how='left')
      df_merged = df_merged.merge(df_products[['productId', 'productName']], on='productId', how='left')

      purchase_counts_ = df_merged.groupby(['Cluster', 'productName'])['items'].sum().reset_index()
      purchase_counts_['Cluster'] = purchase_counts_['Cluster'].astype(str)

      fig_ = px.bar(
        purchase_counts_,
        x='items',
        y='productName',
        color='Cluster',
        orientation='h',
        title='Top Purchased Products by Segment',
        labels={'items': 'Total Items Purchased', 'productName': 'Product'},
        barmode='group',
        height=600
      )

      fig_.update_layout(yaxis={'categoryorder':'total ascending'})
      st.plotly_chart(fig_)

    elif selected_method_ == 'Hierarchical':
      df_rfm_hc_ = df_rfm_hc_.reset_index()
      df_merged = df_transactions[df_transactions['productId'].isin(top_n_product_ids_)]
      df_merged = df_merged.merge(df_rfm_hc_[['Member_number', 'Cluster']], on='Member_number', how='left')
      df_merged = df_merged.merge(df_products[['productId', 'productName']], on='productId', how='left')

      purchase_counts_ = df_merged.groupby(['Cluster', 'productName'])['items'].sum().reset_index()
      purchase_counts_['Cluster'] = purchase_counts_['Cluster'].astype(str)

      fig_ = px.bar(
        purchase_counts_,
        x='items',
        y='productName',
        color='Cluster',
        orientation='h',
        title='Top Purchased Products by Segment',
        labels={'items': 'Total Items Purchased', 'productName': 'Product'},
        barmode='group',
        height=600
      )

      fig_.update_layout(yaxis={'categoryorder':'total ascending'})
      st.plotly_chart(fig_)

pg = st.navigation([
    st.Page(page_business_problem, title='Vấn Đề Kinh Doanh'),
    st.Page(page_eda, title='Tổng Quan Dữ Liệu'),
    st.Page(page_analysis_pred, title='Phân Tích & Dự Đoán'),
    st.Page(page_eval_report, title='Đánh giá & Báo cáo'),
    st.Page(page_recommendation, title='Khuyến nghị')
  ],
  position='top'
)

pg.run()
