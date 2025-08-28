import torch
import torch.nn as nn
import torch.nn.functional as F


# class State_Detection(nn.Module):

#     def __init__(self, num_classes,  num_T, hidden, dropout_rate):
#         # input_size: 1 x EEG channel x datapoint
#         super(State_Detection, self).__init__()
#         self.num_T = num_T
#         self.dropout_1 = nn.Dropout(dropout_rate)
#         self.dropout_2 = nn.Dropout(dropout_rate)
#         self.lstm_1 = nn.Sequential(
#             nn.LSTM(256, 256, 1, batch_first=True))
#         self.lstm_2 = nn.Sequential(
#             nn.LSTM(256, 512, 1, batch_first=True))
#         self.block1_1 = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=num_T, kernel_size=(1,8), stride=(1,8)),
#             nn.BatchNorm2d(num_T),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=0.2)
#             )
#         self.block1_2 = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=num_T, kernel_size=(1,8), stride=(1,8)),
#             nn.BatchNorm2d(num_T),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=0.2)
#             )
#         self.block1_3 = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=num_T, kernel_size=(1,2), stride=(1,2)),
#             nn.BatchNorm2d(num_T),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=0.2)
#             )
#         self.block2_1 = nn.Sequential(
#             nn.Conv2d(in_channels=num_T, out_channels=num_T*2, kernel_size=(1,4), stride=(1,4)),
#             nn.BatchNorm2d(num_T*2),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=0.1)
#             )
#         self.block2_2 = nn.Sequential(
#             nn.Conv2d(in_channels=num_T, out_channels=num_T*2, kernel_size=(1,4), stride=(1,4)),
#             nn.BatchNorm2d(num_T*2),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=0.1)
#             )
#         self.block2_3 = nn.Sequential(
#             nn.Conv2d(in_channels=num_T, out_channels=num_T*2, kernel_size=(1,4), stride=(1,4)),
#             nn.BatchNorm2d(num_T*2),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=0.1)
#             )
#         self.block3_1 = nn.Sequential(
#             nn.Conv2d(in_channels=num_T*2, out_channels=num_T*3, kernel_size=(1,4), stride=(1,4)),
#             nn.BatchNorm2d(num_T*3),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
#             # nn.Dropout2d(p=0.2)
#             )
#         self.block3_2 = nn.Sequential(
#             nn.Conv2d(in_channels=num_T*2, out_channels=num_T*3, kernel_size=(1,4), stride=(1,4)),
#             nn.BatchNorm2d(num_T*3),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
#             # nn.Dropout2d(p=0.2)
#             )
#         self.block3_3 = nn.Sequential(
#             nn.Conv2d(in_channels=num_T*2, out_channels=num_T*3, kernel_size=(1,4), stride=(1,4)),
#             nn.BatchNorm2d(num_T*3),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
#             # nn.Dropout2d(p=0.2)
#             )
#         self.block4_1 = nn.Sequential(
#             nn.Conv2d(in_channels=num_T*3, out_channels=num_T*4, kernel_size=(1,4), stride=(1,4)),
#             nn.BatchNorm2d(num_T*4),
#             nn.ReLU(inplace=True)
#             # nn.Dropout2d(p=0.5)
#             )
#         self.block4_2 = nn.Sequential(
#             nn.Conv2d(in_channels=num_T*3, out_channels=num_T*4, kernel_size=(1,4), stride=(1,4)),
#             nn.BatchNorm2d(num_T*4),
#             nn.ReLU(inplace=True)
#             # nn.Dropout2d(p=0.5)
#             )
#         self.block4_3 = nn.Sequential(
#             nn.Conv2d(in_channels=num_T*3, out_channels=num_T*4, kernel_size=(1,4), stride=(1,4)),
#             nn.BatchNorm2d(num_T*4),
#             nn.ReLU(inplace=True)
#             # nn.Dropout2d(p=0.5)
#             )
#         self.block5_1 = nn.Sequential(
#             nn.Conv2d(in_channels=num_T*4, out_channels=num_T*4, kernel_size=(2,1), stride=(2,1)),
#             nn.BatchNorm2d(num_T*4),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=0.2)
#             )
#         self.block5_2 = nn.Sequential(
#             nn.Conv2d(in_channels=num_T*4, out_channels=num_T*4, kernel_size=(2,1), stride=(2,1)),
#             nn.BatchNorm2d(num_T*4),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=0.2)
#             )

#         self.fc = nn.Sequential(
#             nn.Linear(num_T*32, hidden*16),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(hidden*16, num_classes)
#         )
#     def forward(self, F, x_batch):

#         T = nn.functional.interpolate(x_batch, size=(4,64*4), mode='bilinear', align_corners=False)
#         x_batch = self.block1_1(x_batch)
#         F = self.block1_2(F)
#         T = self.block1_3(T)

#         T = T + F
#         x_batch = x_batch + F

#         x_batch = self.block2_1(x_batch)
#         F = self.block2_2(F)
#         T = self.block2_3(T)
#         T = T * F
#         x_batch = x_batch * F

#         x_batch = self.block3_1(x_batch)
#         F = self.block3_2(F)
#         T = self.block3_3(T)
#         T = T + F
#         x_batch = x_batch + F

#         x_batch = self.block4_1(x_batch)
#         F = self.block4_2(F)
#         T = self.block4_3(T)

#         x_batch = self.block5_1(x_batch)
#         T = self.block5_2(T)

#         x_batch = x_batch.reshape(x_batch.size(0), -1)
#         T = T.reshape(T.size(0), -1)

#         final_fea = torch.cat((x_batch, T), dim=1)
#         final_fea = final_fea.reshape(final_fea.size(0), 1, -1)

#         final_fea,_ = self.lstm_1(final_fea)
#         final_fea,_ = self.lstm_2(final_fea)
#         final_fea = final_fea.reshape(final_fea.size(0), -1)

#         out = self.fc(final_fea)
#         return out

class SELayer(nn.Module):
    # NEW: 轻量通道注意力
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class State_Detection(nn.Module):

    def __init__(self, num_classes, num_T, hidden, dropout_rate):
        super(State_Detection, self).__init__()
        self.num_T = num_T
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)

        # LSTM 不变：input_size=256, hidden=(256->512)
        self.lstm_1 = nn.Sequential(
            nn.LSTM(256, 256, 1, batch_first=True))
        self.lstm_2 = nn.Sequential(
            nn.LSTM(256, 512, 1, batch_first=True))

        # NEW: LSTM 前的归一化与 dropout
        self.pre_lstm_ln = nn.LayerNorm(256)
        self.post_lstm_do = nn.Dropout(dropout_rate)

        # ====== 原有 CNN 模块保持不变 ======
        self.block1_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=num_T, kernel_size=(1,8), stride=(1,8)),
            nn.BatchNorm2d(num_T),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)
        )
        self.block1_2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=num_T, kernel_size=(1,8), stride=(1,8)),
            nn.BatchNorm2d(num_T),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)
        )
        self.block1_3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=num_T, kernel_size=(1,2), stride=(1,2)),
            nn.BatchNorm2d(num_T),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)
        )
        self.block2_1 = nn.Sequential(
            nn.Conv2d(in_channels=num_T, out_channels=num_T*2, kernel_size=(1,4), stride=(1,4)),
            nn.BatchNorm2d(num_T*2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )
        self.block2_2 = nn.Sequential(
            nn.Conv2d(in_channels=num_T, out_channels=num_T*2, kernel_size=(1,4), stride=(1,4)),
            nn.BatchNorm2d(num_T*2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )
        self.block2_3 = nn.Sequential(
            nn.Conv2d(in_channels=num_T, out_channels=num_T*2, kernel_size=(1,4), stride=(1,4)),
            nn.BatchNorm2d(num_T*2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )
        self.block3_1 = nn.Sequential(
            nn.Conv2d(in_channels=num_T*2, out_channels=num_T*3, kernel_size=(1,4), stride=(1,4)),
            nn.BatchNorm2d(num_T*3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )
        self.block3_2 = nn.Sequential(
            nn.Conv2d(in_channels=num_T*2, out_channels=num_T*3, kernel_size=(1,4), stride=(1,4)),
            nn.BatchNorm2d(num_T*3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )
        self.block3_3 = nn.Sequential(
            nn.Conv2d(in_channels=num_T*2, out_channels=num_T*3, kernel_size=(1,4), stride=(1,4)),
            nn.BatchNorm2d(num_T*3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )
        self.block4_1 = nn.Sequential(
            nn.Conv2d(in_channels=num_T*3, out_channels=num_T*4, kernel_size=(1,4), stride=(1,4)),
            nn.BatchNorm2d(num_T*4),
            nn.ReLU(inplace=True)
        )
        self.block4_2 = nn.Sequential(
            nn.Conv2d(in_channels=num_T*3, out_channels=num_T*4, kernel_size=(1,4), stride=(1,4)),
            nn.BatchNorm2d(num_T*4),
            nn.ReLU(inplace=True)
        )
        self.block4_3 = nn.Sequential(
            nn.Conv2d(in_channels=num_T*3, out_channels=num_T*4, kernel_size=(1,4), stride=(1,4)),
            nn.BatchNorm2d(num_T*4),
            nn.ReLU(inplace=True)
        )
        self.block5_1 = nn.Sequential(
            nn.Conv2d(in_channels=num_T*4, out_channels=num_T*4, kernel_size=(2,1), stride=(2,1)),
            nn.BatchNorm2d(num_T*4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)
        )
        self.block5_2 = nn.Sequential(
            nn.Conv2d(in_channels=num_T*4, out_channels=num_T*4, kernel_size=(2,1), stride=(2,1)),
            nn.BatchNorm2d(num_T*4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)
        )

        # NEW: SE 注意力（通道级）
        self.se_x = SELayer(num_T*4)
        self.se_t = SELayer(num_T*4)

        # 全连接保持不变（假设你原先 num_T*32 == 512）
        self.fc = nn.Sequential(
            nn.Linear(num_T*32, hidden*16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden*16, num_classes)
        )

        # NEW: 初始化
        self._init_weights()

    def _init_weights(self):
        # 卷积/线性
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # LSTM
        def init_lstm(lstm):
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)
        init_lstm(self.lstm_1[0])
        init_lstm(self.lstm_2[0])

    def forward(self, F_in, x_batch):
        # ====== 原有三支流与融合 ======
        T = F.interpolate(x_batch, size=(4, 64*4), mode='bilinear', align_corners=False)

        x_batch = self.block1_1(x_batch)
        F_in   = self.block1_2(F_in)
        T      = self.block1_3(T)

        T = T + F_in
        x_batch = x_batch + F_in

        x_batch = self.block2_1(x_batch)
        F_in   = self.block2_2(F_in)
        T      = self.block2_3(T)

        T = T * F_in
        x_batch = x_batch * F_in

        x_batch = self.block3_1(x_batch)
        F_in   = self.block3_2(F_in)
        T      = self.block3_3(T)

        T = T + F_in
        x_batch = x_batch + F_in

        x_batch = self.block4_1(x_batch)
        F_in   = self.block4_2(F_in)
        T      = self.block4_3(T)

        x_batch = self.block5_1(x_batch)
        T      = self.block5_2(T)

        # NEW: 通道注意力
        x_batch = self.se_x(x_batch)
        T      = self.se_t(T)

        # ====== 展平 & 构建序列 ======
        x_batch = x_batch.reshape(x_batch.size(0), -1)  # (B, Nx)
        T = T.reshape(T.size(0), -1)                    # (B, Nt)

        final_fea = torch.cat((x_batch, T), dim=1)      # (B, N)
        # 变成 (B, S, 256)：必要时在最后一维做零填充到 256 的整数倍
        N = final_fea.size(1)
        pad = (-N) % 256
        if pad:
            final_fea = F.pad(final_fea, (0, pad))
        S = final_fea.size(1) // 256
        final_fea = final_fea.view(final_fea.size(0), S, 256)

        # NEW: LSTM 前层归一化
        final_fea = self.pre_lstm_ln(final_fea)

        # ====== LSTM 真·时序建模 ======
        final_fea, _ = self.lstm_1(final_fea)  # (B, S, 256)
        final_fea, _ = self.lstm_2(final_fea)  # (B, S, 512)

        # NEW: 取最后时间步 (保持原先 512 维输入到 fc，不改输出维度)
        final_fea = final_fea[:, -1, :]        # (B, 512)
        final_fea = self.post_lstm_do(final_fea)

        out = self.fc(final_fea)
        return out