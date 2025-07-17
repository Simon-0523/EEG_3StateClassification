import torch
import torch.nn as nn


class State_Detection(nn.Module):

    def __init__(self, num_classes,  num_T, hidden, dropout_rate):
        # input_size: 1 x EEG channel x datapoint
        super(State_Detection, self).__init__()
        self.num_T = num_T
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.lstm_1 = nn.Sequential(
            nn.LSTM(256, 256, 1, batch_first=True))
        self.lstm_2 = nn.Sequential(
            nn.LSTM(256, 512, 1, batch_first=True))
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
            # nn.Dropout2d(p=0.2)
            )
        self.block3_2 = nn.Sequential(
            nn.Conv2d(in_channels=num_T*2, out_channels=num_T*3, kernel_size=(1,4), stride=(1,4)),
            nn.BatchNorm2d(num_T*3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
            # nn.Dropout2d(p=0.2)
            )
        self.block3_3 = nn.Sequential(
            nn.Conv2d(in_channels=num_T*2, out_channels=num_T*3, kernel_size=(1,4), stride=(1,4)),
            nn.BatchNorm2d(num_T*3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
            # nn.Dropout2d(p=0.2)
            )
        self.block4_1 = nn.Sequential(
            nn.Conv2d(in_channels=num_T*3, out_channels=num_T*4, kernel_size=(1,4), stride=(1,4)),
            nn.BatchNorm2d(num_T*4),
            nn.ReLU(inplace=True)
            # nn.Dropout2d(p=0.5)
            )
        self.block4_2 = nn.Sequential(
            nn.Conv2d(in_channels=num_T*3, out_channels=num_T*4, kernel_size=(1,4), stride=(1,4)),
            nn.BatchNorm2d(num_T*4),
            nn.ReLU(inplace=True)
            # nn.Dropout2d(p=0.5)
            )
        self.block4_3 = nn.Sequential(
            nn.Conv2d(in_channels=num_T*3, out_channels=num_T*4, kernel_size=(1,4), stride=(1,4)),
            nn.BatchNorm2d(num_T*4),
            nn.ReLU(inplace=True)
            # nn.Dropout2d(p=0.5)
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

        self.fc = nn.Sequential(
            nn.Linear(num_T*32, hidden*16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden*16, num_classes)
        )
    def forward(self, F, x_batch):

        T = nn.functional.interpolate(x_batch, size=(4,64*4), mode='bilinear', align_corners=False)
        x_batch = self.block1_1(x_batch)
        F = self.block1_2(F)
        T = self.block1_3(T)

        T = T + F
        x_batch = x_batch + F

        x_batch = self.block2_1(x_batch)
        F = self.block2_2(F)
        T = self.block2_3(T)
        T = T * F
        x_batch = x_batch * F

        x_batch = self.block3_1(x_batch)
        F = self.block3_2(F)
        T = self.block3_3(T)
        T = T + F
        x_batch = x_batch + F

        x_batch = self.block4_1(x_batch)
        F = self.block4_2(F)
        T = self.block4_3(T)

        x_batch = self.block5_1(x_batch)
        T = self.block5_2(T)

        x_batch = x_batch.reshape(x_batch.size(0), -1)
        T = T.reshape(T.size(0), -1)

        final_fea = torch.cat((x_batch, T), dim=1)
        final_fea = final_fea.reshape(final_fea.size(0), 1, -1)

        final_fea,_ = self.lstm_1(final_fea)
        final_fea,_ = self.lstm_2(final_fea)
        final_fea = final_fea.reshape(final_fea.size(0), -1)

        out = self.fc(final_fea)
        return out