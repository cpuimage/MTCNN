

#include "mtcnn.h"


bool cmpScore(Bbox lsh, Bbox rsh) {
    return lsh.score < rsh.score;
}


MTCNN::MTCNN(const string &model_path) {

    std::vector<std::string> param_files = {
            model_path + "/det1.param",
            model_path + "/det2.param",
            model_path + "/det3.param"
    };

    std::vector<std::string> bin_files = {
            model_path + "/det1.bin",
            model_path + "/det2.bin",
            model_path + "/det3.bin"
    };

    Pnet.load_param(param_files[0].data());
    Pnet.load_model(bin_files[0].data());
    Rnet.load_param(param_files[1].data());
    Rnet.load_model(bin_files[1].data());
    Onet.load_param(param_files[2].data());
    Onet.load_model(bin_files[2].data());
}

MTCNN::MTCNN(const std::vector<std::string> param_files, const std::vector<std::string> bin_files) {
    Pnet.load_param(param_files[0].data());
    Pnet.load_model(bin_files[0].data());
    Rnet.load_param(param_files[1].data());
    Rnet.load_model(bin_files[1].data());
    Onet.load_param(param_files[2].data());
    Onet.load_model(bin_files[2].data());
}


MTCNN::~MTCNN() {
    Pnet.clear();
    Rnet.clear();
    Onet.clear();
}

void MTCNN::SetMinFace(int minSize) {
    minsize = minSize;
}

void MTCNN::generateBbox(ncnn::Mat score, ncnn::Mat location, std::vector<Bbox> &boundingBox_, float scale) {
    const int stride = 2;
    const int cellsize = 12;
    //score p
    float *p = score.channel(1);
    Bbox bbox = {0};
    float inv_scale = 1.0f / scale;
    for (int row = 0; row < score.h; row++) {
        for (int col = 0; col < score.w; col++) {
            if (*p > threshold[0]) {
                bbox.score = *p;
                bbox.x1 = lround((stride * col + 1) * inv_scale);
                bbox.y1 = lround((stride * row + 1) * inv_scale);
                bbox.x2 = lround((stride * col + 1 + cellsize) * inv_scale);
                bbox.y2 = lround((stride * row + 1 + cellsize) * inv_scale);
                bbox.area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1);
                const int index = row * score.w + col;
                for (int channel = 0; channel < 4; channel++) {
                    bbox.regreCoord[channel] = location.channel(channel)[index];
                }
                boundingBox_.push_back(bbox);
            }
            p++;
        }
    }
}

void MTCNN::nms(std::vector<Bbox> &boundingBox_, const float overlap_threshold, string modelname) {
    if (boundingBox_.empty()) {
        return;
    }
    sort(boundingBox_.begin(), boundingBox_.end(), cmpScore);
    float IOU = 0;
    float maxX = 0;
    float maxY = 0;
    float minX = 0;
    float minY = 0;
    std::vector<int> vecPick;
    size_t pickCount = 0;
    std::multimap<float, int> vScores;
    const size_t num_boxes = boundingBox_.size();
    vecPick.resize(num_boxes);
    for (int i = 0; i < num_boxes; ++i) {
        vScores.insert(std::pair<float, int>(boundingBox_[i].score, i));
    }
    while (!vScores.empty()) {
        int last = vScores.rbegin()->second;
        vecPick[pickCount] = last;
        pickCount += 1;
        for (auto it = vScores.begin(); it != vScores.end();) {
            int it_idx = it->second;
            maxX = std::max(boundingBox_.at(it_idx).x1, boundingBox_.at(last).x1);
            maxY = std::max(boundingBox_.at(it_idx).y1, boundingBox_.at(last).y1);
            minX = std::min(boundingBox_.at(it_idx).x2, boundingBox_.at(last).x2);
            minY = std::min(boundingBox_.at(it_idx).y2, boundingBox_.at(last).y2);
            //maxX1 and maxY1 reuse 
            maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
            maxY = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
            //IOU reuse for the area of two bbox
            IOU = maxX * maxY;
            if (modelname == ("Union"))
                IOU = IOU / (boundingBox_.at(it_idx).area + boundingBox_.at(last).area - IOU);
            else if (modelname == ("Min")) {
                IOU = IOU / ((boundingBox_.at(it_idx).area < boundingBox_.at(last).area) ? boundingBox_.at(it_idx).area
                                                                                         : boundingBox_.at(last).area);
            }
            if (IOU > overlap_threshold) {
                it = vScores.erase(it);
            } else {
                it++;
            }
        }
    }

    vecPick.resize(pickCount);
    std::vector<Bbox> tmp_;
    tmp_.resize(pickCount);
    for (int i = 0; i < pickCount; i++) {
        tmp_[i] = boundingBox_[vecPick[i]];
    }
    boundingBox_ = tmp_;
}

void MTCNN::refine(std::vector<Bbox> &vecBbox, const int &height, const int &width, bool square) {
    if (vecBbox.empty()) {
        cout << "Bbox is empty!!" << endl;
        return;
    }
    float bbw = 0, bbh = 0, maxSide = 0;
    float h = 0, w = 0;
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    for (auto &it : vecBbox) {
        bbw = it.x2 - it.x1 + 1;
        bbh = it.y2 - it.y1 + 1;
        x1 = it.x1 + it.regreCoord[0] * bbw;
        y1 = it.y1 + it.regreCoord[1] * bbh;
        x2 = it.x2 + it.regreCoord[2] * bbw;
        y2 = it.y2 + it.regreCoord[3] * bbh;
        if (square) {
            w = x2 - x1 + 1;
            h = y2 - y1 + 1;
            maxSide = (h > w) ? h : w;
            x1 += (w - maxSide) * 0.5f;
            y1 += (h - maxSide) * 0.5f;
            it.x2 = lround(x1 + maxSide - 1);
            it.y2 = lround(y1 + maxSide - 1);
            it.x1 = lround(x1);
            it.y1 = lround(y1);
        }
        //boundary check
        if (it.x1 < 0)it.x1 = 0;
        if (it.y1 < 0)it.y1 = 0;
        if (it.x2 > width)it.x2 = width - 1;
        if (it.y2 > height)it.y2 = height - 1;

        it.area = (it.x2 - it.x1) * (it.y2 - it.y1);
    }
}

void MTCNN::PNet() {
    firstBbox.clear();
    float minl = img_w < img_h ? img_w : img_h;
    float m = (float) MIN_DET_SIZE / minsize;
    minl *= m;
    float factor = pre_facetor;
    std::vector<float> scales;
    while (minl > MIN_DET_SIZE) {
        scales.push_back(m);
        minl *= factor;
        m = m * factor;
    }
    for (float scale : scales) {
        int hs = (int) ceil(img_h * scale);
        int ws = (int) ceil(img_w * scale);
        ncnn::Mat in;
        resize_bilinear(img, in, ws, hs);
        ncnn::Extractor ex = Pnet.create_extractor();
        ex.set_light_mode(true);
        ex.input("data", in);
        ncnn::Mat score, location;
        ex.extract("prob1", score);
        ex.extract("conv4-2", location);
        std::vector<Bbox> boundingBox;
        generateBbox(score, location, boundingBox, scale);
        nms(boundingBox, nms_threshold[0]);
        firstBbox.insert(firstBbox.end(), boundingBox.begin(), boundingBox.end());
        boundingBox.clear();
    }
}

void MTCNN::RNet() {
    secondBbox.clear();
    for (auto &it : firstBbox) {
        ncnn::Mat tempIm;
        copy_cut_border(img, tempIm, it.y1, img_h - it.y2, it.x1, img_w - it.x2);
        ncnn::Mat in;
        resize_bilinear(tempIm, in, 24, 24);
        ncnn::Extractor ex = Rnet.create_extractor();
        ex.set_light_mode(true);
        ex.input("data", in);
        ncnn::Mat score, bbox;
        ex.extract("prob1", score);
        ex.extract("conv5-2", bbox);
        if ((float) score[1] > threshold[1]) {
            for (int channel = 0; channel < 4; channel++) {
                it.regreCoord[channel] = (float) bbox[channel];
            }
            it.area = (it.x2 - it.x1) * (it.y2 - it.y1);
            it.score = score.channel(1)[0];
            secondBbox.push_back(it);
        }
    }
}

void MTCNN::ONet() {
    thirdBbox.clear();
    for (auto &it : secondBbox) {
        ncnn::Mat tempIm;
        copy_cut_border(img, tempIm, it.y1, img_h - it.y2, it.x1, img_w - it.x2);
        ncnn::Mat in;
        resize_bilinear(tempIm, in, 48, 48);
        ncnn::Extractor ex = Onet.create_extractor();
        ex.set_light_mode(true);
        ex.input("data", in);
        ncnn::Mat score, bbox, keyPoint;
        ex.extract("prob1", score);
        ex.extract("conv6-2", bbox);
        ex.extract("conv6-3", keyPoint);
        if ((float) score[1] > threshold[2]) {
            for (int channel = 0; channel < 4; channel++) {
                it.regreCoord[channel] = (float) bbox[channel];
            }
            it.area = (it.x2 - it.x1) * (it.y2 - it.y1);
            it.score = score.channel(1)[0];
            for (int num = 0; num < 5; num++) {
                (it.ppoint)[num] = it.x1 + (it.x2 - it.x1) * keyPoint[num];
                (it.ppoint)[num + 5] = it.y1 + (it.y2 - it.y1) * keyPoint[num + 5];
            }
            thirdBbox.push_back(it);
        }
    }
}

void MTCNN::detect(ncnn::Mat &img_, std::vector<Bbox> &finalBbox_) {
    img = img_;
    img_w = img.w;
    img_h = img.h;
    img.substract_mean_normalize(mean_vals, norm_vals);
    PNet();
    //the first stage's nms
    if (firstBbox.empty()) return;
    nms(firstBbox, nms_threshold[0]);
    refine(firstBbox, img_h, img_w, true);
    printf("firstBbox_.size()=%d\n", (int) firstBbox.size());
    //second stage
    RNet();
    printf("secondBbox_.size()=%d\n", (int) secondBbox.size());
    if (secondBbox.empty())
        return;
    nms(secondBbox, nms_threshold[1]);
    refine(secondBbox, img_h, img_w, true);
    //third stage 
    ONet();
    printf("thirdBbox_.size()=%d\n", (int) thirdBbox.size());
    if (thirdBbox.empty())
        return;
    refine(thirdBbox, img_h, img_w, true);
    nms(thirdBbox, nms_threshold[2], "Min");
    finalBbox_ = thirdBbox;
}
