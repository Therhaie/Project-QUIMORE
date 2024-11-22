import cv2
import numpy as np

def detect_yellow_ball(image):
    """
    Détecte la balle jaune dans une image.
    
    Arguments:
    image -- image au format OpenCV (BGR)
    
    Sorties:
    (bool, tuple, float, float, float) -- Détection, position, largeur, hauteur, surface
    """
    # Convertir l'image en espace de couleur HSV pour isoler la couleur jaune
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Définir les limites pour la couleur jaune en HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    # Créer un masque pour détecter les pixels jaunes
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Trouver les contours dans l'image segmentée
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialiser les variables de sortie
    detected = False
    position = (0, 0)
    width = height = area = 0
    
    if contours:
        # Sélectionner le contour ayant la plus grande surface
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculer la surface
        area = cv2.contourArea(largest_contour)
        
        # Vérifier la circularité
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter == 0:
            return detected, position, width, height, area
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        # Ajuster un seuil de circularité pour s'assurer que c'est une balle (environ circulaire)
        if circularity > 0.7:  # Seuil de circularité ajustable selon les besoins
            # Récupérer la boîte englobante pour obtenir les dimensions et la position
            x, y, width, height = cv2.boundingRect(largest_contour)
            
            # Calculer la position comme barycentre
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                position = (cx, cy)
            
            detected = True
            
    return detected, position, width, height, area

# Exemple d'utilisation
# Charger une image de test
# image = cv2.imread('chemin/vers/image.jpg')
# detected, position, width, height, area = detect_yellow_ball(image)
# print("Détection:", detected)
# print("Position:", position)
# print("Largeur:", width)
# print("Hauteur:", height)
# print("Surface:", area)
